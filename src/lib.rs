#![cfg_attr(docsrs, feature(doc_cfg))]
//! domi provides abstractions and utilities for
//! [domain-list-community](https://github.com/v2fly/domain-list-community)
//! data source.
//!
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
//!
//!     let content = fs::read_to_string(data_root.join(BASE)).unwrap();
//!
//!     let mut entries = Entries::parse(BASE, content.lines());
//!
//!     while let Some(i) = entries.next_include() {
//!         if entries.is_included(i.target()) {
//!             continue;
//!         }
//!         let include = fs::read_to_string(data_root.join(i.target())).unwrap();
//!         entries.parse_include(i.target(), include.lines());
//!     }
//!
//!     println!("{:?}", entries)
//! }
//! ```

#[cfg(feature = "prost")]
pub mod geosite;
#[cfg(feature = "serde")]
pub mod srs;

mod interner;

use std::{
    collections::{BTreeMap, HashSet, VecDeque},
    fmt::Display,
    ops::Deref,
    rc::Rc,
    str::Lines,
};

use cfg_if::cfg_if;

use crate::interner::{
    BasePool, InternId, InternIdHasher, PackedAttr, PoolGuard, intern, maybe_intern,
};

#[cfg(feature = "smallvec")]
const SMALL_VEC_STACK_SIZE: usize = 4;

cfg_if! {
    if #[cfg(feature = "ahash")] {
        use ::ahash::RandomState as Hasher;
    } else if #[cfg(feature = "rustc-hash")] {
        use ::rustc_hash::FxBuildHasher as Hasher;
    } else {
        use ::std::collections::hash_map::RandomState as Hasher;
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

/// The type of an [`Entry`].
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
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Entry {
    pub kind: Kind,
    pub value: Rc<str>,
    pub attrs: Rc<[Rc<str>]>,
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
            Some(unsafe { Self::new_unchecked(s) })
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

cfg_if! {
    if #[cfg(feature = "smallvec")] {
        type AttrSlice = ::smallvec::SmallVec<[Rc<str>; SMALL_VEC_STACK_SIZE]>;
    } else {
        type AttrSlice = ::std::boxed::Box<[Rc<str>]>;
    }
}

impl Entry {
    #[inline]
    fn ptr_eq(&self, other: &Self) -> bool {
        self.kind == other.kind
            && Rc::ptr_eq(&self.value, &other.value)
            && Rc::ptr_eq(&self.attrs, &other.attrs)
    }

    pub fn parse_line(line: OneLine) -> Option<Self> {
        Self::parse_line_inner::<false>(line)
    }

    #[inline(always)]
    fn parse_line_inner<const USE_POOL: bool>(line: OneLine) -> Option<Self> {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            return None;
        }

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

        let mut parts = value.split_ascii_whitespace();

        let value = parts.next().map(|s| {
            if matches!(kind, Kind::Include) {
                maybe_intern!(USE_POOL, s, Base)
            } else {
                maybe_intern!(USE_POOL, s, DomainValue)
            }
        })?;

        let attrs: AttrSlice = parts
            .filter_map(|s| {
                s.strip_prefix('@')
                    .map(|s| maybe_intern!(USE_POOL, s, Attr))
            })
            .collect();
        let attrs = maybe_intern!(USE_POOL, attrs.as_ref(), AttrSlice);

        Some(Self { kind, value, attrs })
    }
}

#[derive(Debug, Default)]
struct BaseEntries {
    normal: Vec<Entry>,
    includes: Vec<Include>,
    next_include: usize,
    queued: bool,
}

/// An include directive discovered during parsing.
///
/// Instances of this type are returned by
/// [`Entries::next_include`] and [`Entries::drain_includes`].
#[derive(Debug, Clone)]
pub struct Include {
    target: Rc<str>,
    attrs: AttrFilterSlice,
}

impl Include {
    /// Returns the target specified by this include directive.
    ///
    /// The returned string is the raw target specified by the input.
    /// This library does not resolve or interpret it. Callers are
    /// responsible for mapping it to the appropriate resource, such
    /// as by joining it with a base path or performing an
    /// application-specific lookup.
    #[inline]
    pub fn target(&self) -> &str {
        &self.target
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
    bases: BTreeMap<Rc<str>, BaseEntries>,
    parsed_id: HashSet<InternId, InternIdHasher>,
    include_queue: VecDeque<Rc<str>>,
    _pg: PoolGuard,
}

impl Entries {
    #[inline]
    pub fn parse(base: &str, content: Lines) -> Self {
        let mut ret = Self::default();
        ret.parse_include(base, content);
        ret
    }

    fn parse_extend_inner(&mut self, id: Rc<str>, content: Lines) {
        let node = self.bases.entry(id.clone()).or_default();

        content
            .filter_map(|line| {
                // Safety:
                // `line` comes from `Lines`, which guarantees no `\n` or `\r`.
                Entry::parse_line_inner::<true>(unsafe { OneLine::new_unchecked(line) })
            })
            .for_each(|entry| Self::extend_for_each(entry, node, &id, &mut self.include_queue));
    }

    /// Parses and appends entries to the specified base.
    ///
    /// Unlike [`Entries::parse_include`], this function does not track whether
    /// the base has already been parsed. Calling it multiple times with the same
    /// base appends additional entries.
    #[inline]
    pub fn parse_extend(&mut self, current: &str, content: Lines) {
        let id = intern!(current, Base);
        self.parse_extend_inner(id, content);
    }

    /// Returns whether `candidate` has already been included.
    ///
    /// This is a performance helper only.
    /// [`Entries::parse_include`] already avoids duplicate insertion, so
    /// checking beforehand is unnecessary unless include parsing itself is
    /// expensive (such as network-backed IO).
    pub fn is_included(&self, candidate: &str) -> bool {
        let Some(id) = BasePool::base_id(candidate) else {
            return false;
        };

        self.parsed_id.contains(&id)
    }

    /// Parses and appends entries to the specified base if it has not already
    /// been parsed.
    ///
    /// This function is intended for processing [`Include`] targets. Subsequent
    /// calls with the same base are ignored.
    #[inline]
    pub fn parse_include(&mut self, current: &str, content: Lines) {
        let id = intern!(current, Base);

        if !self
            .parsed_id
            // Safety: This uses `intern!()` macro, it's guaranteed return a interned `Rc<T>`
            .insert(unsafe { InternId::from_interned(&id) })
        {
            return;
        };

        self.parse_extend_inner(id, content);
    }

    fn extend_for_each(
        entry: Entry,
        node: &mut BaseEntries,
        id: &Rc<str>,
        include_queue: &mut VecDeque<Rc<str>>,
    ) {
        if matches!(entry.kind, Kind::Domain(_)) {
            node.normal.push(entry);
            return;
        }

        let map = |attr: &Rc<str>| {
            if let Some(attr) = attr.strip_prefix('-') {
                PackedAttr::new(
                    // Safety: This uses `intern!()` macro, it's guaranteed return a interned `Rc<T>`
                    unsafe { InternId::from_interned(&intern!(attr, Attr)) },
                    false,
                )
            } else {
                // Safety: `attr` is from `Entry::parse_line_inner::<true>`,
                // which guarantees to use intern pool
                PackedAttr::new(unsafe { InternId::from_interned(attr) }, true)
            }
        };

        node.includes.push(Include {
            target: entry.value,
            attrs: entry.attrs.iter().map(map).collect(),
        });

        if !node.queued {
            node.queued = true;
            include_queue.push_back(id.clone());
        }
    }

    /// Returns an iterator over all bases.
    ///
    /// Bases are ordered by their [`Ord`] implementation.
    pub fn bases(&self) -> impl Iterator<Item = Rc<str>> + use<> {
        let bases: Box<[_]> = self.bases.keys().cloned().collect();
        bases.into_iter()
    }

    /// Returns `true` if the specified base exists.
    #[inline]
    pub fn contains_base<S: AsRef<str>>(&self, base: S) -> bool {
        self.bases.contains_key(base.as_ref())
    }

    /// Removes the specified base, including all entries and includes.
    ///
    /// Returns `true` if the base existed.
    #[inline]
    pub fn remove_base<S: AsRef<str>>(&mut self, base: S) -> bool {
        self.bases.remove(base.as_ref()).is_some()
    }

    fn pop_helper<F>(&mut self, mut cmp: F) -> bool
    where
        F: FnMut(&Entry) -> bool,
    {
        for node in self.bases.values_mut() {
            if let Some(pos) = node.normal.iter().position(&mut cmp) {
                node.normal.swap_remove(pos);
                return true;
            }
        }

        false
    }

    /// Removes a [`Entry`] from the list.
    #[inline]
    pub fn pop(&mut self, entry: &Entry) -> bool {
        self.pop_helper(|e| e == entry)
    }

    /// Equivalent to [`Entries::pop`], but compares entries using
    /// [`Rc::ptr_eq`] where applicable.
    #[inline]
    pub fn pop_ptr_eq(&mut self, entry: &Entry) -> bool {
        self.pop_helper(|e| e.ptr_eq(entry))
    }

    /// Take and returns the inner [`Vec<Entry>`][Entry]
    pub fn take(&mut self) -> Vec<Entry> {
        let mut out = Vec::new();

        for base in self.bases.values_mut() {
            out.append(&mut base.normal);
        }

        out
    }

    /// Returns a snapshot iterator of current includes.
    ///
    /// Note:
    /// This iterator is **not live**. Newly added includes (e.g. via
    /// [`parse_include`][Self::parse_include]) will **not** appear in the iterator returned by
    /// this call.
    ///
    /// Calling this method advances the internal include cursor to the end, so
    /// all currently pending includes are considered visited.
    ///
    /// To process includes incrementally, call [`Entries::drain_includes`] repeatedly.
    pub fn drain_includes(&mut self) -> impl Iterator<Item = Include> + use<> {
        let mut out = Vec::new();

        while let Some(id) = self.include_queue.pop_front() {
            let Some(node) = self.bases.get_mut(&id) else {
                continue;
            };

            out.extend(node.includes[node.next_include..].iter().cloned());

            node.next_include = node.includes.len();
            node.queued = false;
        }

        out.into_iter()
    }

    /// Returns the next pending include, advancing the internal include cursor.
    ///
    /// Returns [`None`] if no unvisited includes remain.
    pub fn next_include(&mut self) -> Option<Include> {
        while let Some(id) = self.include_queue.pop_front() {
            let Some(node) = self.bases.get_mut(&id) else {
                continue;
            };

            let include = node.includes[node.next_include].clone();
            node.next_include += 1;

            if node.next_include == node.includes.len() {
                node.queued = false;
            } else {
                self.include_queue.push_back(id);
            }

            return Some(include);
        }

        None
    }

    #[inline]
    fn cap(&self) -> usize {
        self.bases.values().map(|b| b.normal.len()).sum()
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
    ///   - [`AttrFilter::Has`] and [`AttrFilter::Lacks`] must NOT appear together for the same key.
    ///     If they do, the filter set is considered invalid and matches nothing.
    ///
    ///
    /// Returns [`None`] if no domains are selected (i.e., the result is empty),
    /// or if `base` was never seen during parsing.
    pub fn flatten(
        &mut self,
        base: &str,
        attr_filters: Option<&[AttrFilter]>,
    ) -> Option<FlatDomains> {
        if self.bases.is_empty() {
            return None;
        }
        let base = BasePool::base_ref(base)?;

        // Convert `AttrFilter` into an internal `Rc` version for fast lookup.
        // After intern lookup, comparisons in flatten/filter use `Rc::ptr_eq`
        // instead of string-by-string comparison, which significantly improves
        // performance on hot paths with many domains.
        //
        // Note: If a caller provides an attribute not already in the AttrPool,
        // it will be interned on-the-fly, incurring a one-time interning cost
        // for that specific filter.
        let afs_map = |f: &AttrFilter| match f {
            AttrFilter::Has(s) => {
                // Safety: This uses `intern!()` macro, it's guaranteed return a interned `Rc<T>`
                PackedAttr::new(unsafe { InternId::from_interned(&intern!(*s, Attr)) }, true)
            }
            AttrFilter::Lacks(s) => PackedAttr::new(
                // Safety: This uses `intern!()` macro, it's guaranteed return a interned `Rc<T>`
                unsafe { InternId::from_interned(&intern!(*s, Attr)) },
                false,
            ),
        };

        let attr_filters: Option<AttrFilterSlice> =
            attr_filters.map(|afs| afs.iter().map(afs_map).collect());

        let mut visited = Vec::with_capacity(8);
        let mut active_filters = Vec::with_capacity(8);

        let mut flattened = Vec::with_capacity(self.cap());

        self.flatten_recursive(&base, &mut visited, &mut flattened, &mut active_filters);

        flattened
            .retain(|entry| flatten::filter_matches_all(&entry.attrs, attr_filters.as_deref()));

        flattened.sort_unstable_by(flatten::sort);
        flattened.dedup_by(flatten::dedup);

        flatten::polish(&mut flattened);

        if flattened.is_empty() {
            return None;
        };

        Some(FlatDomains {
            base,
            inner: flattened,
        })
    }

    fn flatten_recursive(
        &self,
        base: &Rc<str>,
        visited: &mut Vec<InternId>,
        flattened: &mut Vec<Entry>,
        active_filters: &mut Vec<PackedAttr>,
    ) {
        // Safety: base is from `BasePool::base_ref()?` at `Self::flatten()`
        let intern_id = unsafe { InternId::from_interned(base) };

        if visited.contains(&intern_id) {
            return;
        }
        visited.push(intern_id);

        let Some(node) = self.bases.get(base) else {
            visited.pop();
            return;
        };

        let filters = if active_filters.is_empty() {
            None
        } else {
            Some(active_filters.as_slice())
        };

        for entry in &node.normal {
            if flatten::filter_matches_all(&entry.attrs, filters) {
                flattened.push(entry.clone());
            }
        }

        for include in &node.includes {
            let old_len = active_filters.len();

            active_filters.extend(include.attrs.iter().copied());

            self.flatten_recursive(&include.target, visited, flattened, active_filters);

            active_filters.truncate(old_len);
        }

        visited.pop();
    }
}

mod flatten {
    use std::{cmp::Ordering, rc::Rc};

    use super::*;

    #[inline]
    pub(crate) fn sort(a: &Entry, b: &Entry) -> Ordering {
        a.kind
            .cmp(&b.kind)
            .reverse()
            .then_with(|| a.value.cmp(&b.value))
    }

    #[inline]
    pub(crate) fn dedup(a: &mut Entry, b: &mut Entry) -> bool {
        a.ptr_eq(b)
    }

    #[inline]
    pub(crate) fn filter_matches_all(
        candidate: &[Rc<str>],
        attr_filters: Option<&[PackedAttr]>,
    ) -> bool {
        match attr_filters {
            None => true,

            Some([]) => candidate.is_empty(),

            Some(attr_filters) => attr_filters
                .iter()
                .all(|packed_attr| filter_matches(candidate, packed_attr)),
        }
    }

    #[inline]
    pub(crate) fn filter_matches(candidate: &[Rc<str>], packed_attr: &PackedAttr) -> bool {
        if packed_attr.tag() {
            candidate
                .iter()
                .any(|attr| attr.as_ptr().addr() == packed_attr.addr())
        } else {
            candidate
                .iter()
                .all(|attr| attr.as_ptr().addr() != packed_attr.addr())
        }
    }

    pub(crate) fn polish(entries: &mut Vec<Entry>) {
        let mut domains: HashSet<Rc<str>, Hasher> = HashSet::default();

        domains.reserve(entries.len());

        for entry in entries.iter() {
            if matches!(entry.kind, Kind::Domain(DomainKind::Suffix)) {
                domains.insert(entry.value.clone());
            }
        }
        entries.retain(|entry| match entry.kind {
            Kind::Domain(DomainKind::Regex) | Kind::Domain(DomainKind::Keyword) => true,

            Kind::Domain(DomainKind::Suffix) | Kind::Domain(DomainKind::Full) => {
                if !entry.attrs.is_empty() {
                    return true;
                }

                is_not_redundant(entry, &domains)
            }

            Kind::Include => unreachable!(),
        });
    }

    fn is_not_redundant(entry: &Entry, domains: &HashSet<Rc<str>, Hasher>) -> bool {
        let mut domain: &str = &entry.value;

        if matches!(entry.kind, Kind::Domain(DomainKind::Full)) && domains.contains(domain) {
            return false;
        }

        while let Some((_, rest)) = domain.split_once('.') {
            domain = rest;
            if domains.contains(domain) {
                return false;
            }
        }

        true
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
        type AttrFilterSlice = ::std::boxed::Box<[PackedAttr]>;
    }
}

/// Domain entries flattened by [`Entries::flatten`]
#[derive(Debug, Clone)]
pub struct FlatDomains {
    base: Rc<str>,
    inner: Vec<Entry>,
}

impl FlatDomains {
    #[inline]
    fn new_inner<I, F>(base: &str, iter: I, cmp: F) -> Self
    where
        I: IntoIterator<Item = Entry>,
        F: FnMut(&mut Entry, &mut Entry) -> bool,
    {
        let base = Rc::from(base);
        let mut entries: Vec<_> = iter.into_iter().collect();
        entries.retain(|e| matches!(e.kind, Kind::Domain(_)));
        entries.sort_by(flatten::sort);
        entries.dedup_by(cmp);
        Self {
            base,
            inner: entries,
        }
    }

    /// Creates a [`FlatDomains`] from an iterator of [`Entry`].
    ///
    /// Entries are compared by value rather than interned identity,
    /// allowing equivalent entries from different interning contexts
    /// to deduplicate correctly.
    pub fn new<I>(base: &str, iter: I) -> Self
    where
        I: IntoIterator<Item = Entry>,
    {
        Self::new_inner(base, iter, |a, b| a == b)
    }

    /// Equivalent to [`FlatDomains::new`], but compares entries using
    /// [`Rc::ptr_eq`] where applicable.
    pub fn new_ptr_eq<I>(base: &str, iter: I) -> Self
    where
        I: IntoIterator<Item = Entry>,
    {
        Self::new_inner(base, iter, |a, b| a.ptr_eq(b))
    }

    /// Returns the target base of this flattened result.
    ///
    /// This is the base passed to [`Entries::flatten`], not necessarily the
    /// original base of every contained [`Entry`].
    #[inline]
    pub fn base(&self) -> &str {
        &self.base
    }

    /// Consumes [`self`] and returns the underlying [`Vec<Entry>`][Entry].
    pub fn into_vec(self) -> Vec<Entry> {
        self.inner
    }

    /// inner [`Vec<Entry>`][Entry] will be [`Vec::split_off`]
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

#[cfg(test)]
const BASE: &str = "base";

#[cfg(test)]
mod sort_predictable;

#[cfg(test)]
mod tests {
    use std::assert_matches;

    use super::*;

    #[test]
    fn accepts_normal_str() {
        let s = "hello world";
        let line = OneLine::new(s);

        assert!(line.is_some());
        assert_eq!(&*line.unwrap(), s);
    }

    #[test]
    fn rejects_lf() {
        let s = "hello\nworld";
        assert!(OneLine::new(s).is_none());
    }

    #[test]
    fn rejects_cr() {
        let s = "hello\rworld";
        assert!(OneLine::new(s).is_none());
    }

    #[test]
    fn parse_line_combinations() {
        let _pg = PoolGuard::acquire();
        let attr_combos: [&[&str]; _] = [&[], &["attr1"], &["attr1", "attr2"]];

        let kinds = [
            Kind::Domain(DomainKind::Suffix),
            Kind::Domain(DomainKind::Full),
            Kind::Domain(DomainKind::Keyword),
            Kind::Domain(DomainKind::Regex),
            Kind::Include,
        ];

        for attrs in attr_combos {
            for kind in kinds {
                let mut line = format!("{}:example.com", kind);
                for attr in attrs.iter() {
                    line.push_str(" @");
                    line.push_str(attr);
                }
                let line = OneLine::new(&line).unwrap();

                let entry = Entry::parse_line_inner::<true>(line);

                let attrs: AttrSlice = attrs.iter().map(|s| intern!(*s, Attr)).collect();
                let attrs = intern!(attrs.as_ref(), AttrSlice);

                let expected_domain = Some(Entry {
                    kind,
                    value: intern!("example.com", DomainValue),
                    attrs,
                });

                assert_eq!(entry, expected_domain, "line: {}", line);
            }
        }
    }

    #[test]
    fn entries_take() {
        let mut entries = Entries::parse(
            BASE,
            "\
        include:something # includes won't be taken
        example.com
        "
            .lines(),
        );

        let taken = entries.take();

        assert_eq!(taken.len(), 1);

        // include remains
        assert_eq!(entries.drain_includes().size_hint(), (1, Some(1)))
    }

    #[test]
    fn pop_domain() {
        let mut entries = Entries::parse(BASE, "example.com".lines());
        let entry = &Entries::parse(BASE, "example.com".lines()).take()[0];
        assert!(entries.pop(entry));
    }

    #[test]
    fn flatten_domains() {
        let content = "\
            domain:example.com
            full:example.com       # will be polished out
            full:full.example.com  # will be polished out
            keyword:keyword
            full:full.com
            domain:domain.full.com # will stay
        ";

        let mut entries = Entries::parse(BASE, content.lines());

        let flat = entries.flatten(BASE, None).unwrap();
        let flat_domains = flat.into_vec();

        // len
        assert_eq!(flat_domains.len(), 4);

        // reversed kind ord
        assert_matches!(flat_domains[0].kind, Kind::Domain(DomainKind::Keyword));
        assert_matches!(flat_domains[1].kind, Kind::Domain(DomainKind::Full));
        assert_matches!(flat_domains[2].kind, Kind::Domain(DomainKind::Suffix));
        assert_matches!(flat_domains[3].kind, Kind::Domain(DomainKind::Suffix));

        // lexicographic order
        assert_eq!(&*flat_domains[2].value, "domain.full.com");
        assert_eq!(&*flat_domains[3].value, "example.com");
    }

    #[test]
    fn flatten_domains_take_next() {
        let content = "\
            domain:domain
            full:full
            keyword:keyword
            regexp:regexp
        ";

        let mut entries = Entries::parse(BASE, content.lines());

        let mut flat = entries.flatten(BASE, None).unwrap();

        let kinds = [
            DomainKind::Suffix,
            DomainKind::Full,
            DomainKind::Keyword,
            DomainKind::Regex,
        ];

        let mut i = 0;

        while let Some(e) = flat.take_next() {
            assert_eq!(e.len(), 1);
            assert_eq!(e[0].kind, Kind::Domain(kinds[i]));
            i += 1;
        }

        assert_eq!(i, 4);
    }

    #[test]
    fn dedup() {
        let content = "\
            keyword:keyword
            keyword:keyword # dedup
            full:full
            full:full @attr1 # no dedup
        ";

        let mut entries = Entries::parse(BASE, content.lines());

        let flat = entries.flatten(BASE, None).unwrap().into_vec();

        assert_eq!(flat.len(), 3);

        assert_matches!(flat[0].kind, Kind::Domain(DomainKind::Keyword));
        assert_matches!(flat[1].kind, Kind::Domain(DomainKind::Full));
        assert_matches!(flat[2].kind, Kind::Domain(DomainKind::Full));

        assert_eq!(&*flat[1].value, "full");
        assert_eq!(flat[1].value, flat[2].value);

        assert_ne!(flat[1].attrs, flat[2].attrs);
        assert_eq!(&*flat[2].attrs, ["attr1".into()]);
    }

    #[test]
    fn flat_from_iter() {
        let flat = FlatDomains::new(
            BASE,
            [
                parse_helper("keyword:b"),
                parse_helper("include:ignored"),
                parse_helper("keyword:a"),
                parse_helper("keyword:a"), // dedup
            ],
        )
        .into_vec();

        assert_eq!(flat.len(), 2);
        assert_eq!(&*flat[0].value, "a");
        assert_eq!(&*flat[1].value, "b");
    }

    fn attr_test_helper(content: &[&str], expected: Entry) {
        const BASE: &str = "content_1";

        let mut entries = Entries::parse(BASE, content[0].lines());

        let mut next = 1;

        while let Some(i) = entries.next_include() {
            entries.parse_include(i.target(), content[next].lines());
            next += 1;
        }

        let flat = entries.flatten(BASE, None).unwrap().into_vec();

        assert_eq!(flat.len(), 1);

        assert_eq!(flat[0], expected);
    }

    #[test]
    fn attr_recursive() {
        let content = &[
            "\
            include:content_2 @attr1
        ",
            "\
            include:content_3
        ",
            "\
            content_3 @attr1
            ignored
        ",
        ];

        attr_test_helper(
            content,
            Entry {
                kind: Kind::Domain(DomainKind::Suffix),
                value: "content_3".into(),
                attrs: ["attr1".into()].into(),
            },
        )
    }

    #[test]
    fn negative_attr() {
        let content = &[
            "\
                include:content_2 @-attr2
        ",
            "\
            content_2 @attr1
            ignored @attr2
        ",
        ];

        attr_test_helper(
            content,
            Entry {
                kind: Kind::Domain(DomainKind::Suffix),
                value: "content_2".into(),
                attrs: ["attr1".into()].into(),
            },
        );
    }

    fn parse_helper(s: &str) -> Entry {
        Entry::parse_line(OneLine::new(s).unwrap()).unwrap()
    }

    #[test]
    fn entry_to_flat() {
        // random order, sort
        let entries = [
            parse_helper("regexp:entry_4"),
            parse_helper("entry_1"),
            parse_helper("keyword:entry_3"),
            parse_helper("full:entry_2"),
            parse_helper("full:entry_2"), // dedup
        ];

        let mut flat = FlatDomains::new(BASE, entries);

        assert_eq!(flat.inner.len(), 4);

        assert_eq!(flat.take_next().unwrap()[0], parse_helper("entry_1"));
        assert_eq!(flat.take_next().unwrap()[0], parse_helper("full:entry_2"));
        assert_eq!(
            flat.take_next().unwrap()[0],
            parse_helper("keyword:entry_3")
        );
        assert_eq!(flat.take_next().unwrap()[0], parse_helper("regexp:entry_4"));
    }

    #[test]
    fn base_operation() {
        let mut entries = Entries::parse("base0", "".lines());

        assert!(entries.contains_base("base0"));

        assert!(entries.remove_base("base0"));

        assert!(!entries.contains_base("base0"));
    }

    #[test]
    fn include_paired_with_base_operation() {
        let mut entries = Entries::parse("base0", "include:base1".lines());

        assert!(entries.contains_base("base0"));

        assert!(entries.remove_base("base0"));

        assert_matches!(entries.next_include(), None);

        assert!(!entries.contains_base("base0"));

        //

        let mut entries = Entries::parse("base0", "include:base1".lines());

        assert!(entries.contains_base("base0"));

        assert!(entries.remove_base("base0"));

        let drain: Box<[_]> = entries.drain_includes().collect();
        assert_eq!(drain.len(), 0);

        assert!(!entries.contains_base("base0"));

        //

        let mut entries = Entries::parse("base0", "include:base1".lines());
        entries.parse_include("base1", "include:base2".lines());

        assert!(entries.contains_base("base0"));

        assert!(entries.remove_base("base0"));

        assert_eq!(entries.next_include().unwrap().target(), "base2");

        assert!(!entries.contains_base("base0"));
    }
}
