#![cfg_attr(docsrs, feature(doc_cfg))]
//! domi provides abstractions and utilities for
//! [domain-list-community](https://github.com/v2fly/domain-list-community)
//! data source.
//!
//! ## Example
//! ```rust,no_run
//! use std::{fs, path::Path};
//!
//! use domi::{srs::Rule, Entries};
//!
//! const BASE: &str = "alphabet";
//!
//! fn main() {
//!     let data_root = Path::new("data");
//!     let content = fs::read_to_string(data_root.join(BASE)).unwrap();
//!     let mut entries = Entries::parse(BASE, content.lines());
//!     while let Some(i) = entries.next_include() {
//!         let include = fs::read_to_string(data_root.join(i.as_ref())).unwrap();
//!         entries.parse_extend(BASE, include.lines());
//!     }
//!     // expect: domain_keyword: Some(["fitbit", "google"])
//!     // change the `Some(&[])` to something else can alter behavier,
//!     // see crate::Entries
//!     println!("{:?}", Rule::from(entries.flatten(BASE, Some(&[])).unwrap()))
//! }
//! ```

#[cfg(feature = "prost")]
pub mod geosite;
#[cfg(feature = "serde")]
pub mod srs;

use std::{
    cell::{Cell, RefCell},
    collections::{BTreeSet, HashSet, VecDeque},
    fmt::Display,
    hash::Hash,
    marker::PhantomData,
    mem,
    ops::Deref,
    panic::{AssertUnwindSafe, catch_unwind},
    rc::Rc,
    str::Lines,
};

use cfg_if::cfg_if;

cfg_if! {
    if #[cfg(feature = "ahash")] {
        use ::ahash::RandomState as Hasher;
    } else if #[cfg(feature = "rustc-hash")] {
        use ::rustc_hash::FxBuildHasher as Hasher;
    } else {
        use ::std::collections::hash_map::RandomState as Hasher;
    }
}

struct Interner<T: Eq + Hash + ?Sized> {
    set: Option<HashSet<Rc<T>, Hasher>>,
}

impl<T: Eq + Hash + ?Sized> Interner<T> {
    fn new() -> Self {
        Self { set: None }
    }

    fn initialize(&mut self) {
        self.set = Some(HashSet::default())
    }

    fn intern(&mut self, s: Rc<T>) -> Rc<T> {
        let set = self
            .set
            .as_mut()
            .expect("intern pool not initialized; missing PoolGuard");
        if let Some(v) = set.get(&s) {
            v.clone()
        } else {
            set.insert(s.clone());
            s
        }
    }

    fn intern_ref(&self, value: &T) -> Option<Rc<T>> {
        let set = self.set.as_ref()?;
        set.get(value).cloned()
    }

    fn clear(&mut self) {
        self.set = None;
    }
}

macro_rules! define_pool {
    ($name:ident, $ty:ty) => {
        ::paste::paste! {
            thread_local! {
                static [< $name:snake:upper _POOL >]: RefCell<Interner<$ty>> = RefCell::new(Interner::new());
            }
            struct [< $name Pool >];
            impl [< $name Pool >] {
                fn initialize() {
                    [< $name:snake:upper _POOL >].with(|p| p.borrow_mut().initialize())
                }
                fn [< $name:snake >](value: Rc<$ty>) -> Rc<$ty> {
                    [< $name:snake:upper _POOL >].with(|p| p.borrow_mut().intern(value))
                }
                fn [< $name:snake _ref >](value: &$ty) -> Option<Rc<$ty>> {
                    [< $name:snake:upper _POOL >].with(|p| p.borrow().intern_ref(value))
                }
                fn clear() {
                    [< $name:snake:upper _POOL >].with(|p| p.borrow_mut().clear())
                }
            }
        }
    };
}

define_pool!(Base, str);
define_pool!(Attr, str);
define_pool!(DomainValue, str);
define_pool!(AttrSlice, [Rc<str>]);

macro_rules! maybe_intern {
    ($use_pool:expr, $s:expr, $name:ident) => {{
        if !$use_pool {
            Rc::from($s)
        } else {
            intern!($s, $name)
        }
    }};
}

macro_rules! intern {
    ($s:expr, $name:ident) => {
        ::paste::paste! {
            [<$name Pool>]::[<$name:snake _ref>]($s.as_ref())
                .unwrap_or_else(|| [<$name Pool>]::[<$name:snake>](Rc::from($s)))
        }
    };
}

thread_local! {
    static POOL_USED_COUNT: Cell<isize> = const { Cell::new(0) };
}

type NotSyncNorSend = PhantomData<Rc<()>>;

#[derive(Debug)]
struct PoolGuard {
    _marker: NotSyncNorSend,
}

impl PoolGuard {
    fn acquire() -> Self {
        let n = POOL_USED_COUNT.get();
        if n <= 0 {
            POOL_USED_COUNT.set(1);
            #[cfg(debug_assertions)]
            if n < 0 {
                dbg!("POOL_USED_COUNT underflow", n);
            }
            Self::clear_pools();
            AttrPool::initialize();
            BasePool::initialize();
            DomainValuePool::initialize();
            AttrSlicePool::initialize();
        } else if n == isize::MAX {
            panic!("Pool is poisoned due to previous panic in Drop");
        } else {
            POOL_USED_COUNT.set(n + 1);
        }
        Self {
            _marker: NotSyncNorSend::default(),
        }
    }

    fn clear_pools() {
        AttrPool::clear();
        BasePool::clear();
        DomainValuePool::clear();
        AttrSlicePool::clear();
    }
}

impl Default for PoolGuard {
    fn default() -> Self {
        Self::acquire()
    }
}

impl Drop for PoolGuard {
    fn drop(&mut self) {
        let n = POOL_USED_COUNT.get() - 1;
        if n <= 0 {
            POOL_USED_COUNT.set(0);
            if catch_unwind(AssertUnwindSafe(Self::clear_pools)).is_err() {
                POOL_USED_COUNT.set(isize::MAX);
            };
            #[cfg(debug_assertions)]
            if n < 0 {
                dbg!("POOL_USED_COUNT underflow", n);
            }
        } else {
            POOL_USED_COUNT.set(n);
        }
    }
}

/// Represents the matching behavior
///
/// This corresponds to the prefix of a single domain in the source file
/// (e.g. `domain:`, `full:`, `keyword:`, `regexp:`).
///
/// And if no prefix present, then [`DomainKind::Suffix`] will be chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DomainKind {
    /// This variant's matching prefix is `domain:`.
    Suffix,
    Full,
    Keyword,
    Regex,
}

impl Display for DomainKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match *self {
            DomainKind::Suffix => "domain",
            DomainKind::Full => "full",
            DomainKind::Keyword => "keyword",
            DomainKind::Regex => "regexp",
        })
    }
}

/// Single parsed domain entry
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Domain {
    pub kind: DomainKind,
    pub base: Rc<str>,
    pub value: Rc<str>,
    pub attrs: Rc<[Rc<str>]>,
}

impl Domain {
    #[inline]
    fn rc_matches(&self, base: &Rc<str>, value: &Rc<str>) -> bool {
        Rc::ptr_eq(&self.base, base) && Rc::ptr_eq(&self.value, value)
    }
}

/// A string slice that is guaranteed to be a single line.
///
/// # Invariants
///
/// The contained string must not contain `\n` or `\r`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct OneLine<'a> {
    inner: &'a str,
}

impl<'a> OneLine<'a> {
    /// Creates a [`OneLine`] if the input string contains no line breaks.
    ///
    /// Returns [`None`] if `s` contains `\n` or `\r`.
    pub fn new(s: &'a str) -> Option<Self> {
        if s.find(['\n', '\r']).is_some() {
            None
        } else {
            Some(Self { inner: s })
        }
    }

    /// # Safety
    ///
    /// `s` must not contain `\n` or `\r`.
    pub unsafe fn new_unchecked(s: &'a str) -> Self {
        Self { inner: s }
    }
}

impl<'a> Deref for OneLine<'a> {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.inner
    }
}

impl<'a> AsRef<str> for OneLine<'a> {
    fn as_ref(&self) -> &str {
        self.inner
    }
}

impl<'a> Display for OneLine<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self)
    }
}

#[cfg(test)]
mod one_line {
    use crate::OneLine;

    #[test]
    fn test_accepts_normal_str() {
        let s = "hello world";
        let line = OneLine::new(s);

        assert!(line.is_some());
        assert_eq!(&*line.unwrap(), s);
    }

    #[test]
    fn test_rejects_lf() {
        let s = "hello\nworld";
        assert!(OneLine::new(s).is_none());
    }

    #[test]
    fn test_rejects_cr() {
        let s = "hello\rworld";
        assert!(OneLine::new(s).is_none());
    }
}

cfg_if! {
    if #[cfg(feature = "smallvec")] {
        type AttrSlice = ::smallvec::SmallVec<[Rc<str>; 8]>;
    } else {
        type AttrSlice = Box<[Rc<str>]>;
    }
}

/// Single parsed entry
#[derive(Debug, Clone)]
pub enum Entry {
    Domain(Domain),
    Include(Rc<str>),
}

impl Entry {
    pub fn parse_line(base: &str, line: OneLine) -> Option<Self> {
        Self::parse_line_inner::<false>(base, line)
    }

    #[inline(always)]
    fn parse_line_inner<const USE_POOL: bool>(base: &str, line: OneLine) -> Option<Self> {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            return None;
        }

        let base = maybe_intern!(USE_POOL, base, Base);

        let line = line.split_once('#').map(|(l, _)| l).unwrap_or(line).trim();

        let (kind_str, value) = line.split_once(':').unwrap_or(("domain", line));
        let kind = match kind_str {
            "domain" => DomainKind::Suffix,
            "full" => DomainKind::Full,
            "regexp" => DomainKind::Regex,
            "keyword" => DomainKind::Keyword,
            "include" => return Some(Self::Include(maybe_intern!(USE_POOL, value, Base))),
            _ => unreachable!("unknown domain kind prefix: {kind_str}"),
        };

        let mut parts = value.split_whitespace();

        let value = parts
            .next()
            .map(|s| maybe_intern!(USE_POOL, s, DomainValue))?;

        let attrs: AttrSlice = parts
            .filter_map(|s| {
                s.strip_prefix('@')
                    .map(|s| maybe_intern!(USE_POOL, s, Attr))
            })
            .collect();
        let attrs = maybe_intern!(USE_POOL, attrs.as_ref(), AttrSlice);

        Some(Self::Domain(Domain {
            kind,
            base,
            value,
            attrs,
        }))
    }
}

#[test]
fn test_parse_line_combinations() {
    let _pg = PoolGuard::acquire();
    let bases = ["google", "alphabet"];
    let attr_combos: [&[&str]; _] = [&[], &["attr1"], &["attr1", "attr2"]];

    let kinds = [
        DomainKind::Suffix,
        DomainKind::Full,
        DomainKind::Keyword,
        DomainKind::Regex,
    ];

    for base in bases {
        for attrs in attr_combos {
            for kind in kinds {
                let mut line = format!("{}:example.com", kind);
                for attr in attrs.iter() {
                    line.push_str(" @");
                    line.push_str(attr);
                }
                let line = OneLine::new(&line).unwrap();
                let (domain, include) = match Entry::parse_line_inner::<true>(base, line).unwrap() {
                    Entry::Domain(d) => (Some(d), None),
                    Entry::Include(i) => (None, Some(i)),
                };

                let attrs: AttrSlice = attrs.iter().map(|s| intern!(*s, Attr)).collect();
                let attrs = intern!(attrs.as_ref(), AttrSlice);

                let expected_domain = Some(Domain {
                    kind,
                    base: intern!(base, Base),
                    value: intern!("example.com", DomainValue),
                    attrs,
                });

                assert_eq!(domain, expected_domain, "line: {}", line);
                assert_eq!(include, None, "line: {}", line);
            }
        }
    }
}

/// Parsed entries from source
///
/// This type owns all parsed domains and include directives
/// for a given base.
///
/// While an [`Entries`] value is alive, internal string intern pools are kept alive.
/// They are automatically cleared when the last [`Entries`] is dropped on the thread.
///
/// ## Invariants
///
/// - The include graph is assumed to be **acyclic**.
/// - Cyclic includes are considered invalid data and are **not** checked
///   at runtime.
#[derive(Debug, Default)]
pub struct Entries {
    domains: Vec<Domain>,
    includes: VecDeque<Rc<str>>,
    _pg: PoolGuard,
}

impl Entries {
    pub fn parse(base: &str, content: Lines) -> Self {
        let mut ret = Self::default();
        ret.parse_extend(base, content);
        ret
    }

    /// <div class="warning">
    /// Warning:
    /// This method assumes the include graph to be acyclic.
    /// </div>
    ///
    /// See [`Entries`] for details.
    pub fn parse_extend(&mut self, base: &str, content: Lines) {
        for entry in content.filter_map(|line| {
            // Safety:
            // `line` comes from `Lines`, which guarantees no `\n` or `\r`.
            Entry::parse_line_inner::<true>(base, unsafe { OneLine::new_unchecked(line) })
        }) {
            match entry {
                Entry::Domain(domain) => self.domains.push(domain),
                Entry::Include(include) => self.includes.push_back(include),
            }
        }
    }

    /// Returns a deduplicated set of bases.
    ///
    /// Bases are ordered by their [`Ord`] implementation.
    pub fn bases(&self) -> impl Iterator<Item = Rc<str>> + use<> {
        let btree: BTreeSet<_> = self.domains.iter().map(|d| d.base.clone()).collect();
        btree.into_iter()
    }

    /// Removes a domain from the list by its base and value.
    ///
    /// This looks up the interned references for the provided strings and
    /// removes the first matching domain if it exists.
    pub fn pop_domain(&mut self, base: &str, domain: &str) -> Option<Domain> {
        let base = BasePool::base_ref(base)?;
        let domain = DomainValuePool::domain_value_ref(domain)?;
        let pos = self
            .domains
            .iter()
            .position(|d| d.rc_matches(&base, &domain))?;
        Some(self.domains.swap_remove(pos))
    }

    /// Returns a snapshot iterator of current includes.
    ///
    /// Note:
    /// This iterator is **not live**. Newly added includes (e.g. via
    /// `parse_extend`) will **not** appear in the iterator returned by
    /// this call.
    ///
    /// To process includes incrementally, call [`Entries::drain_includes`] repeatedly.
    /// # Example:
    /// ```rust,no_run
    /// # use std::fs;
    /// # use domi::Entries;
    /// # const BASE: &str = "";
    /// # let mut entries = Entries::parse(BASE, "".lines());
    /// while let Some(i) = entries.drain_includes().next() {
    ///     let include = fs::read_to_string(i.as_ref()).unwrap();
    ///     entries.parse_extend(BASE, include.lines());
    /// }
    /// ```
    ///
    /// <div class="warning">
    /// Warning:
    /// This method assumes the include graph to be acyclic.
    /// </div>
    ///
    /// See [`Entries`] for details.
    pub fn drain_includes(&mut self) -> impl Iterator<Item = Rc<str>> + use<> {
        mem::take(&mut self.includes).into_iter()
    }

    /// Returns and consume one include
    ///
    /// <div class="warning">
    /// Warning:
    /// This method assumes the include graph to be acyclic.
    /// </div>
    ///
    /// See [`Entries`] for details.
    pub fn next_include(&mut self) -> Option<Rc<str>> {
        self.includes.pop_front()
    }

    /// Flatten domains by `base` with optional attribute filters,
    /// then **[`sort`][slice::sort]** and **[`dedup`][Vec::dedup]** the selected domains.
    ///
    /// # Selection rules:
    /// - `attr_filters == None`:
    ///   Selects **all** domains with a matching `base`.
    ///
    /// - `attr_filters == Some(&[])`:
    ///   Selects **only** domains with a matching `base` and **no** attributes.
    ///
    /// - `attr_filters == Some(filters)`:
    ///   Selects domains with a matching `base` that satisfy **all** filters:
    ///   - [`AttrFilter::Has`]: **At least one** of `candidate.attrs` matches the filter value.
    ///   - [`AttrFilter::Lacks`]: **No** `candidate.attrs` matches the filter value.
    ///     This effectively **overrides** any [`AttrFilter::Has`] matches for the same attribute.
    ///
    /// ### Performance
    /// Unlike [`flatten_drain`][Self::flatten_drain], this method retains the original domains
    /// by cloning each selected [`Domain`], which incurs some additional allocation overhead.
    ///
    ///
    /// Returns [`None`] if no domains are selected (i.e., the result is empty),
    /// or if `base` was never seen during parsing.
    pub fn flatten(
        &mut self,
        base: &str,
        attr_filters: Option<&[AttrFilter]>,
    ) -> Option<FlatDomains> {
        self.flatten_inner::<false>(base, attr_filters)
    }

    /// Similar to [`flatten`][Self::flatten], but **drains** selected domains from `self.domains`.
    ///
    /// Only non-selected domains are retained in the original collection. This is generally more
    /// efficient than [`flatten`][Self::flatten] as it moves out domains instead of cloning them.
    pub fn flatten_drain(
        &mut self,
        base: &str,
        attr_filters: Option<&[AttrFilter]>,
    ) -> Option<FlatDomains> {
        self.flatten_inner::<true>(base, attr_filters)
    }

    #[inline(always)]
    fn flatten_inner<const DRAIN: bool>(
        &mut self,
        base: &str,
        attr_filters: Option<&[AttrFilter]>,
    ) -> Option<FlatDomains> {
        if self.domains.is_empty() {
            return None;
        }
        let mut flattened = Vec::with_capacity(self.domains.len());
        let base = BasePool::base_ref(base)?;
        // Convert `AttrFilter` into an internal `Rc` version for fast lookup.
        // After intern lookup, comparisons in flatten/filter use `Rc::ptr_eq`
        // instead of string-by-string comparison, which significantly improves
        // performance on hot paths with many domains.
        //
        // Note: If a caller provides an attribute not already in the AttrPool,
        // it will be interned on-the-fly, incurring a one-time interning cost
        // for that specific filter.
        let attr_filters: Option<AttrFilterSlice> = attr_filters.map(|afs| {
            afs.iter()
                .map(|f| match f {
                    AttrFilter::Has(s) => AttrFilterIntern {
                        id: AttrId::from(intern!(*s, Attr)),
                        has: true,
                    },
                    AttrFilter::Lacks(s) => AttrFilterIntern {
                        id: AttrId::from(intern!(*s, Attr)),
                        has: false,
                    },
                })
                .collect()
        });

        if DRAIN {
            flatten::drain_matches(
                &mut self.domains,
                base,
                attr_filters.as_deref(),
                &mut flattened,
            )
        } else {
            flatten::retain_all(&self.domains, base, attr_filters.as_deref(), &mut flattened)
        }

        if flattened.is_empty() {
            return None;
        };

        flattened.sort_by(|a, b| {
            b.kind
                .cmp(&a.kind) // reverse kind order to enable efficient tail `Vec::drain()`.
                .then_with(|| a.value.cmp(&b.value)) // sort value by dictionary order
        });
        flattened.dedup();
        Some(FlatDomains { inner: flattened })
    }
}

mod flatten {
    use std::rc::Rc;

    use crate::{AttrFilterIntern, Domain};

    #[inline]
    fn should_select(
        candidate: &Domain,
        base: &Rc<str>,
        attr_filters: Option<&[AttrFilterIntern]>,
    ) -> bool {
        if !Rc::ptr_eq(base, &candidate.base) {
            return false;
        }

        match &attr_filters {
            None => true,
            Some([]) => candidate.attrs.is_empty(),
            Some(attr_filters) => attr_filters.iter().all(|attr_filter| {
                if attr_filter.has {
                    candidate
                        .attrs
                        .iter()
                        .any(|attr| attr.as_ptr() == attr_filter.id)
                } else {
                    candidate
                        .attrs
                        .iter()
                        .all(|attr| attr.as_ptr() != attr_filter.id)
                }
            }),
        }
    }

    pub(crate) fn retain_all(
        domains: &[Domain],
        base: Rc<str>,
        attr_filters: Option<&[AttrFilterIntern]>,
        flattened: &mut Vec<Domain>,
    ) {
        domains.iter().for_each(|candidate| {
            if should_select(candidate, &base, attr_filters) {
                flattened.push(candidate.clone());
            }
        });
    }

    pub(crate) fn drain_matches(
        domains: &mut Vec<Domain>,
        base: Rc<str>,
        attr_filters: Option<&[AttrFilterIntern]>,
        flattened: &mut Vec<Domain>,
    ) {
        flattened.extend(domains.extract_if(.., |candidate| {
            should_select(candidate, &base, attr_filters)
        }));
    }
}

/// Filtering behavior. Used by [`Entries::flatten`]
pub enum AttrFilter<'a> {
    Has(&'a str),
    Lacks(&'a str),
}

cfg_if! {
    if #[cfg(feature = "smallvec")] {
        type AttrFilterSlice = ::smallvec::SmallVec<[AttrFilterIntern; 8]>;
    } else {
        type AttrFilterSlice = Box<[AttrFilterIntern]>;
    }
}

// Discards DST metadata
// saves some stack memory
#[derive(Debug, PartialEq, Eq)]
#[repr(transparent)]
struct AttrId(usize);

impl From<Rc<str>> for AttrId {
    fn from(value: Rc<str>) -> Self {
        Self(value.as_ptr() as usize)
    }
}

impl PartialEq<*const u8> for AttrId {
    fn eq(&self, other: &*const u8) -> bool {
        self.0 == *other as usize
    }
}

impl PartialEq<AttrId> for *const u8 {
    fn eq(&self, other: &AttrId) -> bool {
        *self as usize == other.0
    }
}

#[test]
fn test_attr_id() {
    let _pg = PoolGuard::acquire();
    let a = AttrId::from(intern!(BASE, Attr));
    let b = AttrId::from(intern!(BASE, Attr));
    assert_eq!(a, b);
}

// SAFETY INVARIANTS:
//
// - intern pool guarantees unique allocation per string
// - intern pool lives at least as long as all AttrFilterIntern
// - id is only used for equality comparison
// - id is never dereferenced
struct AttrFilterIntern {
    id: AttrId,
    has: bool,
}

#[cfg(test)]
const BASE: &str = "base";

#[test]
fn test_pop_domain() {
    let mut entries = Entries::parse(BASE, "example.com".lines());
    entries.pop_domain(BASE, "example.com").unwrap();
    assert_eq!(0, entries.domains.len());
}

#[test]
fn test_parse_entries_basic() {
    let content = "\
            # comment line
            domain:example.com @attr1 @attr2
            full:full.example.com
            include:example # trailing comment
        ";

    let mut entries = Entries::parse(BASE, content.lines());

    assert_eq!(entries.domains.len(), 2);
    assert_eq!(entries.domains[0].kind, DomainKind::Suffix);
    assert_eq!(entries.domains[0].value.as_ref(), "example.com");
    assert_eq!(entries.domains[0].attrs.len(), 2);
    assert_eq!(entries.domains[1].kind, DomainKind::Full);
    assert_eq!(entries.domains[1].value.as_ref(), "full.example.com");

    let includes: Box<_> = entries.drain_includes().collect();
    assert_eq!(includes.len(), 1);
    assert_eq!(includes[0].as_ref(), "example");
}

/// Domain entries flattened by [`Entries::flatten`], reversed ordered by [`DomainKind`].
///
/// Domains are grouped by [`DomainKind`] and sorted lexicographically by value.
#[derive(Clone)]
pub struct FlatDomains {
    inner: Vec<Domain>,
}

impl FlatDomains {
    /// Consumes [`self`] and returns the underlying [`Vec<Domain>`][Domain].
    pub fn into_vec(self) -> Vec<Domain> {
        self.inner
    }

    /// inner [`Vec<Domain>`][Domain] will be [`Vec::drain`]
    /// at the next kind index to reduce allocations.
    ///
    /// At most one call per [`DomainKind`] variant (maximum 4 calls).
    pub fn take_next(&mut self) -> Option<(DomainKind, Box<[Rc<str>]>)> {
        let kind = self.inner.last()?.kind;
        let idx = self.inner.partition_point(|d| d.kind != kind);
        let v: Box<[_]> = self.inner.drain(idx..).map(|d| d.value).collect();
        (!v.is_empty()).then_some((kind, v))
    }
}

#[test]
fn test_flatten_domains() {
    let content = "\
            domain:example.com
            full:full.example.com
            keyword:keyword
        ";

    let mut entries = Entries::parse(BASE, content.lines());

    let flat = entries.flatten_drain(BASE, None).unwrap();

    assert!(entries.domains.is_empty());

    let flat_domains = flat.into_vec();
    assert_eq!(flat_domains.len(), 3);
    assert!(flat_domains.iter().any(|d| d.kind == DomainKind::Suffix));
    assert!(flat_domains.iter().any(|d| d.kind == DomainKind::Full));
    assert!(flat_domains.iter().any(|d| d.kind == DomainKind::Keyword));
}

#[test]
fn test_flatten_partial_domains() {
    let content = "\
            domain:example.com
            full:full.example.com
        ";

    let mut entries = Entries::parse(BASE, content.lines());

    let other_base = BasePool::base("other_base".into());
    entries.domains[0].base = other_base.clone();

    let flat = entries.flatten_drain(BASE, None).unwrap();

    let flat_domains = flat.into_vec();
    assert_eq!(flat_domains.len(), 1);
    assert_eq!(flat_domains[0].base.as_ref(), BASE);

    assert_eq!(entries.domains.len(), 1);
    assert_eq!(entries.domains[0].base.as_ptr(), other_base.as_ptr());
}

#[test]
fn test_flatten_domains_take_next() {
    let content = "\
            domain:domain
            full:full
            keyword:keyword
            regexp:regexp
        ";

    let mut entries = Entries::parse(BASE, content.lines());

    let mut flat = entries.flatten_drain(BASE, None).unwrap();

    assert!(entries.domains.is_empty());

    let mut i = 0;
    while flat.take_next().is_some() {
        i += 1;
    }
    assert_eq!(i, 4);
}

#[test]
fn test_dedup() {
    let content = "\
            keyword:keyword
            keyword:keyword # dedup
        ";

    let mut entries = Entries::parse(BASE, content.lines());

    let flat = entries.flatten_drain(BASE, None).unwrap().into_vec();

    assert!(entries.domains.is_empty());

    assert_eq!(flat[0].kind, DomainKind::Keyword);
}

#[cfg(test)]
mod sort_predictable {
    use std::array;

    use crate::{BASE, Entries, FlatDomains};

    const VARIANT_LEN: usize = 6;
    const CONTENS: [&str; VARIANT_LEN] = [
        "full:full\nkeyword:keyword\nregexp:regexp",
        "full:full\nregexp:regexp\nkeyword:keyword",
        "keyword:keyword\nfull:full\nregexp:regexp",
        "keyword:keyword\nregexp:regexp\nfull:full",
        "regexp:regexp\nfull:full\nkeyword:keyword",
        "regexp:regexp\nkeyword:keyword\nfull:full",
    ];

    pub(crate) fn test<T, K, F, C>(mut build: F, mut cmp: C)
    where
        F: FnMut(FlatDomains) -> T,
        K: Ord,
        C: FnMut(&T) -> K,
    {
        let list: [T; VARIANT_LEN] = array::from_fn(|i| {
            let domains = Entries::parse(BASE, CONTENS[i].lines())
                .flatten_drain(BASE, None)
                .unwrap();

            build(domains)
        });

        assert!(list.windows(2).all(|w| cmp(&w[0]) == cmp(&w[1])));
    }
}
