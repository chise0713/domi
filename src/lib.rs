#![cfg_attr(docsrs, feature(doc_cfg))]
//! domi provides abstractions and utilities for
//! [domain-list-community](https://github.com/v2fly/domain-list-community)
//! data source.
//!
//!
//! <div class="warning">
//! Warning:
//! The crate is not updated with official implementation, DO NOT use in production
//! </div>
//!
//! ## Example
//! ```rust,no_run
//! use std::{fs, path::Path};
//!
//! use domi::Entries;
//!
//! const BASE: &str = "alphabet";
//!
//! fn main() {
//!     let data_root = Path::new("data");
//!     let content = fs::read_to_string(data_root.join(BASE)).unwrap();
//!     let mut entries = Entries::parse(BASE, content.lines());
//!     while let Some(i) = entries.next_include() {
//!         let include = fs::read_to_string(data_root.join(i.base.as_ref())).unwrap();
//!         entries.parse_extend(i.base.as_ref(), BASE, include.lines());
//!     }
//!     // expect: domain_keyword: Some(["fitbit", "google"])
//!     // change the `Some(&[])` to something else can alter behavier,
//!     // see crate::Entries
//!     println!("{:?}", entries)
//! }
//! ```

#[cfg(feature = "prost")]
pub mod geosite;
#[cfg(feature = "serde")]
pub mod srs;

use std::{
    cell::{Cell, RefCell},
    cmp::Ordering,
    collections::{BTreeSet, HashSet},
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

#[cfg(feature = "smallvec")]
const SMALL_VEC_STACK_SIZE: usize = 4;

struct Interner<T>
where
    T: Eq + Hash + ?Sized,
{
    set: Option<HashSet<Rc<T>, Hasher>>,
}

impl<T> Interner<T>
where
    T: Eq + Hash + ?Sized,
{
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

// Discards DST metadata
// saves some stack memory
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
struct InternId(usize);

impl InternId {
    /// # Safety
    ///
    /// MUST be Intern-ed by Interner
    #[inline(always)]
    unsafe fn from_interned<T: ?Sized>(value: Rc<T>) -> Self {
        Self(Rc::as_ptr(&value).addr())
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
    ($use_pool:expr, $s:expr, $name:ident) => {
        if !$use_pool {
            Rc::from($s)
        } else {
            intern!($s, $name)
        }
    };
}

macro_rules! intern {
    ($s:expr, $name:ident) => {
        ::paste::paste! {
            crate::[<$name Pool>]::[<$name:snake _ref>]($s.as_ref())
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    Domain(DomainKind),
    Include,
}

impl Display for Kind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::Domain(d) => d.fmt(f),
            Self::Include => f.write_str("include"),
        }
    }
}

/// Single parsed entry
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub kind: Kind,
    pub base: Rc<str>,
    pub value: Rc<str>,
    pub attrs: Rc<[Rc<str>]>,
}

impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        match (&self.kind, &other.kind) {
            (Kind::Include, Kind::Include) => self
                .value
                .cmp(&other.value)
                .then_with(|| self.base.cmp(&other.base))
                .then_with(|| self.attrs.cmp(&other.attrs)),
            (Kind::Include, _) => Ordering::Less,
            (_, Kind::Include) => Ordering::Greater,

            (Kind::Domain(a), Kind::Domain(b)) => a
                .cmp(b)
                .then_with(|| self.value.cmp(&other.value))
                .then_with(|| self.base.cmp(&other.base))
                .then_with(|| self.attrs.cmp(&other.attrs)),
        }
    }
}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
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
        type AttrSlice = ::smallvec::SmallVec<[Rc<str>; SMALL_VEC_STACK_SIZE]>;
    } else {
        type AttrSlice = Box<[Rc<str>]>;
    }
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
            "domain" => Kind::Domain(DomainKind::Suffix),
            "full" => Kind::Domain(DomainKind::Full),
            "regexp" => Kind::Domain(DomainKind::Regex),
            "keyword" => Kind::Domain(DomainKind::Keyword),
            "include" => Kind::Include,
            _ => unimplemented!("unknown domain kind prefix: {kind_str}"),
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

        Some(Self {
            kind,
            base,
            value,
            attrs,
        })
    }
}

#[test]
fn test_parse_line_combinations() {
    let _pg = PoolGuard::acquire();
    let bases = ["google", "alphabet"];
    let attr_combos: [&[&str]; _] = [&[], &["attr1"], &["attr1", "attr2"]];

    let kinds = [
        Kind::Domain(DomainKind::Suffix),
        Kind::Domain(DomainKind::Full),
        Kind::Domain(DomainKind::Keyword),
        Kind::Domain(DomainKind::Regex),
        Kind::Include,
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

                let entry = Entry::parse_line_inner::<true>(base, line);

                let attrs: AttrSlice = attrs.iter().map(|s| intern!(*s, Attr)).collect();
                let attrs = intern!(attrs.as_ref(), AttrSlice);

                let expected_domain = Some(Entry {
                    kind,
                    base: intern!(base, Base),
                    value: intern!("example.com", DomainValue),
                    attrs,
                });

                assert_eq!(entry, expected_domain, "line: {}", line);
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
#[derive(Debug, Default)]
pub struct Entries {
    entries: BTreeSet<Entry>,
    parsed_id: BTreeSet<Rc<str>>,
    _pg: PoolGuard,
}

impl Entries {
    pub fn parse(base: &str, content: Lines) -> Self {
        let mut ret = Self::default();
        ret.parse_extend(base, base, content);
        ret
    }

    pub fn parse_extend(&mut self, current: &str, base: &str, content: Lines) {
        let id = intern!(current, Base);
        if self.parsed_id.contains(&id) {
            return;
        };
        content
            .filter_map(|line| {
                // Safety:
                // `line` comes from `Lines`, which guarantees no `\n` or `\r`.
                Entry::parse_line_inner::<true>(base, unsafe { OneLine::new_unchecked(line) })
            })
            .for_each(|entry| {
                self.entries.insert(entry);
            });
        self.parsed_id.insert(id);
    }

    /// Returns a deduplicated set of bases.
    ///
    /// Bases are ordered by their [`Ord`] implementation.
    pub fn bases(&self) -> impl Iterator<Item = Rc<str>> + use<> {
        let btree: BTreeSet<_> = self.entries.iter().map(|d| d.base.clone()).collect();
        btree.into_iter()
    }

    /// Removes a [`Entry`] from the list.
    pub fn pop(&mut self, entry: &Entry) -> bool {
        self.entries.remove(entry)
    }

    /// Take and returns the inner [`Vec<Entry>`][Entry]
    pub fn take(&mut self) -> Vec<Entry> {
        mem::take(&mut self.entries).into_iter().collect()
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
    ///     let include = fs::read_to_string(i.base.as_ref()).unwrap();
    ///     entries.parse_extend(i.base.as_ref(), BASE, include.lines());
    /// }
    /// ```
    pub fn drain_includes(&mut self) -> impl Iterator<Item = Entry> + use<> {
        let entries = mem::take(&mut self.entries);
        let (includes, others) = entries
            .into_iter()
            .partition(|e| matches!(e.kind, Kind::Include));
        self.entries = others;
        includes.into_iter()
    }

    /// Returns and consume one include
    pub fn next_include(&mut self) -> Option<Entry> {
        let entry = self
            .entries
            .range(..)
            .find(|e| matches!(e.kind, Kind::Include))
            .cloned()?;
        self.entries.remove(&entry).then_some(entry)
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
        if self.entries.is_empty() {
            return None;
        }
        let mut flattened = Vec::with_capacity(self.entries.len());
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
                    AttrFilter::Has(s) => {
                        PackedAttr::new(unsafe { InternId::from_interned(intern!(*s, Attr)) }, true)
                    }
                    AttrFilter::Lacks(s) => PackedAttr::new(
                        unsafe { InternId::from_interned(intern!(*s, Attr)) },
                        false,
                    ),
                })
                .collect()
        });

        if DRAIN {
            flatten::drain_matches(
                &mut self.entries,
                base,
                attr_filters.as_deref(),
                &mut flattened,
            )
        } else {
            flatten::retain_all(&self.entries, base, attr_filters.as_deref(), &mut flattened)
        }

        if flattened.is_empty() {
            return None;
        };

        Some(FlatDomains { inner: flattened })
    }
}

mod flatten {
    use std::{collections::BTreeSet, rc::Rc};

    use crate::{Entry, Kind, PackedAttr};

    #[inline]
    fn should_select(
        candidate: &Entry,
        base: &Rc<str>,
        attr_filters: Option<&[PackedAttr]>,
    ) -> bool {
        if matches!(candidate.kind, Kind::Include) || !Rc::ptr_eq(base, &candidate.base) {
            return false;
        }

        match &attr_filters {
            None => true,
            Some([]) => candidate.attrs.is_empty(),
            Some(attr_filters) => attr_filters.iter().all(|packed_attr| {
                if packed_attr.tag() {
                    candidate
                        .attrs
                        .iter()
                        .any(|attr| attr.as_ptr() as usize == packed_attr.addr())
                } else {
                    candidate
                        .attrs
                        .iter()
                        .all(|attr| attr.as_ptr() as usize != packed_attr.addr())
                }
            }),
        }
    }

    pub(crate) fn retain_all(
        domains: &BTreeSet<Entry>,
        base: Rc<str>,
        attr_filters: Option<&[PackedAttr]>,
        flattened: &mut Vec<Entry>,
    ) {
        domains.iter().for_each(|candidate| {
            if should_select(candidate, &base, attr_filters) {
                flattened.push(candidate.clone());
            }
        });
    }

    pub(crate) fn drain_matches(
        domains: &mut BTreeSet<Entry>,
        base: Rc<str>,
        attr_filters: Option<&[PackedAttr]>,
        flattened: &mut Vec<Entry>,
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
        type AttrFilterSlice = ::smallvec::SmallVec<[PackedAttr; SMALL_VEC_STACK_SIZE]>;
    } else {
        type AttrFilterSlice = Box<[PackedAttr]>;
    }
}

#[repr(transparent)]
struct PackedAttr {
    inner: usize,
}

impl PackedAttr {
    const TAG_MASK: usize = 0x1;
    const ADDR_MASK: usize = !Self::TAG_MASK;

    #[inline(always)]
    pub const fn new(attr_id: InternId, tag: bool) -> Self {
        let InternId(ptr_addr) = attr_id;
        assert!(ptr_addr.is_multiple_of(2));
        Self {
            inner: ptr_addr | tag as usize,
        }
    }

    #[inline(always)]
    pub const fn addr(&self) -> usize {
        self.inner & Self::ADDR_MASK
    }

    #[inline(always)]
    pub const fn tag(&self) -> bool {
        (self.inner & Self::TAG_MASK) != 0
    }
}

#[test]
fn test_packed_attr_logic() {
    let original_addr = 0x12345670_usize;
    let attr_id = InternId(original_addr);

    let packed_true = PackedAttr::new(attr_id, true);
    assert!(packed_true.tag());
    assert_eq!(packed_true.addr(), original_addr,);
    assert_eq!(packed_true.inner, original_addr | 1);

    let packed_false = PackedAttr::new(attr_id, false);
    assert!(!packed_false.tag());
    assert_eq!(packed_false.addr(), original_addr,);
    assert_eq!(packed_false.inner, original_addr);
}

#[test]
fn test_with_real_pointer() {
    let val = Box::new(42);
    let ptr_addr = &*val as *const i32 as usize;

    assert_eq!(ptr_addr & 0x1, 0,);

    let attr_id = InternId(ptr_addr);

    let p1 = PackedAttr::new(attr_id, true);
    let p2 = PackedAttr::new(attr_id, false);

    assert!(p1.tag());
    assert!(!p2.tag());
    assert_eq!(p1.addr(), ptr_addr);
    assert_eq!(p2.addr(), ptr_addr);
}

#[test]
fn test_const_capability() {
    const ADDR: InternId = InternId(0x1000);
    const PACKED: PackedAttr = PackedAttr::new(ADDR, true);

    const { assert!(PACKED.tag()) };
    assert_eq!(PACKED.addr(), 0x1000);
}

#[cfg(test)]
const BASE: &str = "base";

#[test]
fn test_pop_domain() {
    let mut entries = Entries::parse(BASE, "example.com".lines());
    let entry = entries.entries.first().unwrap().clone();
    assert!(entries.pop(&entry));
}

/// Domain entries flattened by [`Entries::flatten`]
#[derive(Clone)]
pub struct FlatDomains {
    inner: Vec<Entry>,
}

impl FlatDomains {
    /// Consumes [`self`] and returns the underlying [`Vec<Domain>`][Domain].
    pub fn into_vec(self) -> Vec<Entry> {
        self.inner
    }

    /// inner [`Vec<Domain>`][Domain] will be [`Vec::split_off`]
    /// at the next kind index to reduce allocations.
    ///
    /// At most one call per [`DomainKind`] variant (maximum 4 calls).
    pub fn take_next(&mut self) -> Option<Box<[Entry]>> {
        let kind = self.inner.last()?.kind;
        let idx = self.inner.partition_point(|d| d.kind != kind);
        let v = self.inner.split_off(idx).into_boxed_slice();
        (!v.is_empty()).then_some(v)
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

    assert!(entries.entries.is_empty());

    let flat_domains = flat.into_vec();
    assert_eq!(flat_domains.len(), 3);
    assert!(
        flat_domains
            .iter()
            .any(|d| d.kind == Kind::Domain(DomainKind::Suffix))
    );
    assert!(
        flat_domains
            .iter()
            .any(|d| d.kind == Kind::Domain(DomainKind::Full))
    );
    assert!(
        flat_domains
            .iter()
            .any(|d| d.kind == Kind::Domain(DomainKind::Keyword))
    );
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

    assert!(entries.entries.is_empty());

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

    assert!(entries.entries.is_empty());

    assert_eq!(flat[0].kind, Kind::Domain(DomainKind::Keyword));
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
        K: Ord,
        F: FnMut(FlatDomains) -> T,
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
