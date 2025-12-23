#![cfg_attr(docsrs, feature(doc_cfg))]
//! domi provides abstractions and utilities for
//! [domain-list-community](https://github.com/v2fly/domain-list-community)
//! data source.
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
//!     let mut entries = Entries::parse(BASE, content.lines()).unwrap();
//!     while let Some(i) = entries.next_include() {
//!         let include = fs::read_to_string(data_root.join(i.as_ref())).unwrap();
//!         entries.parse_extend(include.lines()).unwrap();
//!     }
//!     // expect: domain_keyword: ["fitbit", "google"]
//!     // change the `Some(&[])` to something else can alter behavier,
//!     // see crate::Entries
//!     println!("{:?}", entries.flatten(BASE, Some(&[])).unwrap().dump())
//! }
//! ```

#[cfg(feature = "protobuf")]
pub mod geosite;
#[cfg(feature = "serde")]
pub mod srs;

use std::{
    cell::RefCell,
    collections::{HashSet, VecDeque},
    fmt::Display,
    rc::Rc,
    str::Lines,
};

use rustc_hash::FxBuildHasher;

/// Represents the matching behavier
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
    pub value: Box<str>,
    pub attrs: Box<[Rc<str>]>,
}

struct Interner {
    set: HashSet<Rc<str>, FxBuildHasher>,
}

impl Interner {
    fn new() -> Self {
        Self {
            set: HashSet::with_hasher(FxBuildHasher),
        }
    }

    fn intern(&mut self, s: &str) -> Rc<str> {
        if let Some(v) = self.set.get(s) {
            v.clone()
        } else {
            let v: Rc<str> = Rc::from(s);
            self.set.insert(v.clone());
            v
        }
    }
}

thread_local! {
    static ATTR_POOL: RefCell<Interner> = RefCell::new(Interner::new());
}

struct AttrPool;

impl AttrPool {
    #[inline]
    fn attr(s: &str) -> Rc<str> {
        ATTR_POOL.with(|p| p.borrow_mut().intern(s))
    }
}

thread_local! {
    static BASE_POOL: RefCell<Interner> = RefCell::new(Interner::new());
}

struct BasePool;

impl BasePool {
    #[inline]
    fn base(s: &str) -> Rc<str> {
        BASE_POOL.with(|p| p.borrow_mut().intern(s))
    }
}

/// Single parsed entry
#[derive(Debug, Clone)]
pub enum Entry {
    Domain(Domain),
    Include(Rc<str>),
}

impl Entry {
    pub fn parse_line(line: &str, base: &str) -> Option<Self> {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            return None;
        }

        let line = line.split_once('#').map(|(l, _)| l).unwrap_or(line).trim();

        let (kind_str, value) = line.split_once(':').unwrap_or(("domain", line));
        let kind = match kind_str {
            "domain" => DomainKind::Suffix,
            "full" => DomainKind::Full,
            "regexp" => DomainKind::Regex,
            "keyword" => DomainKind::Keyword,
            "include" => return Some(Self::Include(BasePool::base(value))),
            _ => unreachable!("unknown domain kind prefix: {kind_str}"),
        };

        let mut parts = value.split_whitespace();

        let value = parts.next()?.into();

        let attrs = parts
            .filter(|s| s.starts_with('@'))
            .map(|s| AttrPool::attr(s[1..].into()))
            .collect();

        let base = BasePool::base(base);

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
    let bases = ["google", "alphabet"];
    let attr_combos: [&[&str]; _] = [&[], &["attr1"], &["attr1", "attr2"]];

    let kinds = [
        DomainKind::Suffix,
        DomainKind::Full,
        DomainKind::Keyword,
        DomainKind::Regex,
    ];

    for base in &bases {
        for attrs in &attr_combos {
            for kind in &kinds {
                let mut line = format!("{}:example.com", kind);
                for attr in attrs.iter() {
                    line.push_str(" @");
                    line.push_str(attr);
                }

                let (domain, include) = match Entry::parse_line(&line, base).unwrap() {
                    Entry::Domain(d) => (Some(d), None),
                    Entry::Include(i) => (None, Some(i)),
                };

                let expected_domain = Some(Domain {
                    kind: *kind,
                    base: BasePool::base(base),
                    value: "example.com".into(),
                    attrs: attrs.iter().map(|s| AttrPool::attr(s)).collect(),
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
/// ## Invariants
///
/// - The include graph is assumed to be **acyclic**.
/// - Cyclic includes are considered invalid data and are **not** checked
///   at runtime.
pub struct Entries {
    base: Rc<str>,
    domains: Option<Vec<Domain>>,
    includes: Option<VecDeque<Rc<str>>>,
}

impl Entries {
    pub fn parse(base: &str, content: Lines) -> Option<Self> {
        let mut domains = Vec::new();
        let mut includes = VecDeque::new();
        let base = BasePool::base(base);

        for line in content {
            if let Some(entry) = Entry::parse_line(line, &base) {
                match entry {
                    Entry::Domain(mut domain) => {
                        domain.base = base.clone();
                        domains.push(domain);
                    }
                    Entry::Include(include) => includes.push_back(include),
                }
            }
        }

        Some(Self {
            base,
            domains: Some(domains).filter(|d| !d.is_empty()),
            includes: Some(includes).filter(|i| !i.is_empty()),
        })
    }

    fn set_domains(&mut self, domains: Vec<Domain>) {
        self.domains = Some(domains).filter(|i| !i.is_empty());
    }

    fn set_includes(&mut self, includes: VecDeque<Rc<str>>) {
        self.includes = Some(includes).filter(|i| !i.is_empty());
    }

    /// <div class="warning">
    /// Warning:
    /// This method assumes the include graph to be acyclic.
    /// </div>
    ///
    /// See [`Entries`] for details.
    pub fn parse_extend(&mut self, content: Lines) -> Option<()> {
        let entries = Self::parse(&self.base, content)?;

        if let Some(domains) = entries.domains {
            match self.domains.take() {
                Some(mut d) => {
                    d.extend(domains);
                    self.set_domains(d);
                }
                None => self.set_domains(domains),
            };
        };

        if let Some(includes) = entries.includes {
            match self.includes.take() {
                Some(mut i) => {
                    i.extend(includes);
                    self.set_includes(i);
                }
                None => self.set_includes(includes),
            };
        };

        Some(())
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
    /// # let mut entries = Entries::parse("", "".lines()).unwrap();
    /// while let Some(i) = entries.drain_includes().next() {
    ///     let include = fs::read_to_string(i.as_ref()).unwrap();
    ///     entries.parse_extend(include.lines()).unwrap();
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
        self.includes.take().into_iter().flatten()
    }

    /// <div class="warning">
    /// Warning:
    /// This method assumes the include graph to be acyclic.
    /// </div>
    ///
    /// See [`Entries`] for details.
    pub fn next_include(&mut self) -> Option<Rc<str>> {
        let mut includes = self.includes.take()?;
        let one = includes.pop_front();
        self.set_includes(includes);
        one
    }

    /// Flatten domains by `base` with optional attribute filters.
    ///
    /// Selection rules:
    /// - `attr_filters == None`  
    ///   → select **all** domains with matching `base`
    ///
    /// - `attr_filters == Some(&[])`  
    ///   → select **only** domains with matching `base` and no attributes
    ///
    /// - `attr_filters == Some(filters)`  
    ///   → select domains with matching `base` that satisfy **all** filters:
    ///     - [`AttrFilter::Has`]: **at least one** of `candidate.attrs` equals to variant value
    ///     - [`AttrFilter::Lacks`]: **no `candidate.attrs` equals** to variant value;
    ///       **overrides** [`AttrFilter::Has`] matches
    ///
    /// Selected domains are removed from `self.domains`;
    /// non-selected domains are retained.
    /// Returns `None` if `self.domains` is empty, or if no domains are selected (`flattened` is empty).
    pub fn flatten(
        &mut self,
        base: &str,
        attr_filters: Option<&[AttrFilter]>,
    ) -> Option<FlatDomains> {
        let inner = self.domains.take()?;
        let mut flattened = Vec::with_capacity(inner.len());

        let domains: Vec<_> = inner
            .into_iter()
            .filter_map(|candidate| {
                if base != &*candidate.base {
                    return Some(candidate);
                }

                let select = match attr_filters {
                    None => true,
                    Some([]) => candidate.attrs.is_empty(),
                    Some(attr_filters) => {
                        attr_filters.iter().all(|attr_filter| match attr_filter {
                            AttrFilter::Has(matches) => {
                                candidate.attrs.iter().any(|attr| &**attr == *matches)
                            }
                            AttrFilter::Lacks(matches) => {
                                candidate.attrs.iter().all(|attr| &**attr != *matches)
                            }
                        })
                    }
                };

                if select {
                    flattened.push(candidate);
                    None
                } else {
                    Some(candidate)
                }
            })
            .collect();

        self.set_domains(domains);

        Some(FlatDomains(flattened)).filter(|f| !f.0.is_empty())
    }
}

/// Filtering behavier. Used by [`Entries::flatten`]
pub enum AttrFilter<'a> {
    Has(&'a str),
    Lacks(&'a str),
}

#[cfg(test)]
const BASE: &str = "base";

#[test]
fn test_parse_entries_basic() {
    let content = "\
            # comment line
            domain:example.com @attr1 @attr2
            full:full.example.com
            include:example # trailing comment
        ";

    let mut entries = Entries::parse(BASE, content.lines()).unwrap();

    let domains = entries.domains.take().unwrap();
    assert_eq!(domains.len(), 2);
    assert_eq!(domains[0].kind, DomainKind::Suffix);
    assert_eq!(domains[0].value.as_ref(), "example.com");
    assert_eq!(domains[0].attrs.len(), 2);
    assert_eq!(domains[1].kind, DomainKind::Full);
    assert_eq!(domains[1].value.as_ref(), "full.example.com");

    let includes: Box<_> = entries.drain_includes().collect();
    assert_eq!(includes.len(), 1);
    assert_eq!(includes[0].as_ref(), "example");
}

/// Domain entires dumped by [`FlatDomains::dump`]
#[derive(Debug)]
pub struct Dump {
    pub domain_suffix: Box<[Box<str>]>,
    pub domain: Box<[Box<str>]>,
    pub domain_keyword: Box<[Box<str>]>,
    pub domain_regex: Box<[Box<str>]>,
}

/// Domain entries flattened by [`Entries::flatten`]
///
/// This struct's inner [`Vec`] will always not be [`Vec::is_empty`]
pub struct FlatDomains(Vec<Domain>);

impl FlatDomains {
    pub fn dump(self) -> Dump {
        let mut domains: Vec<Domain> = self.0;
        debug_assert!(!domains.is_empty());

        domains.sort_by(|a, b| {
            b.kind
                .cmp(&a.kind) // kind reversed for `domains.split_off()`
                .then_with(|| a.value.cmp(&b.value)) // sort value by dictionary order
        });
        domains.dedup();

        let mut next = |kind: DomainKind| {
            let idx = domains.partition_point(|d| d.kind != kind);
            domains.split_off(idx)
        };

        let suffix = next(DomainKind::Suffix);
        let full = next(DomainKind::Full);
        let keyword = next(DomainKind::Keyword);
        let regex = next(DomainKind::Regex);
        drop(domains);

        let domain_suffix = suffix.into_iter().map(|d| d.value).collect();
        let domain = full.into_iter().map(|d| d.value).collect();
        let domain_keyword = keyword.into_iter().map(|d| d.value).collect();
        let domain_regex = regex.into_iter().map(|d| d.value).collect();

        Dump {
            domain_suffix,
            domain,
            domain_keyword,
            domain_regex,
        }
    }
}

#[test]
fn test_flatten_domains() {
    let content = "\
            domain:example.com
            full:full.example.com
            keyword:keyword
        ";

    let mut entries = Entries::parse(BASE, content.lines()).unwrap();

    let flat = entries.flatten(BASE, None).unwrap();

    assert!(entries.domains.is_none());

    let flat_domains = &flat.0;
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

    let mut entries = Entries::parse(BASE, content.lines()).unwrap();

    let other_base = BasePool::base("other_base");
    if let Some(domains) = entries.domains.as_mut() {
        domains[0].base = other_base.clone();
    }

    let flat = entries.flatten(BASE, None).unwrap();

    let flat_domains = &flat.0;
    assert_eq!(flat_domains.len(), 1);
    assert_eq!(flat_domains[0].base.as_ref(), BASE);

    let remaining = entries.domains.as_ref().unwrap();
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0].base.as_ptr(), other_base.as_ptr());
}

#[test]
fn test_dedup() {
    let content = "\
            keyword:keyword
            keyword:keyword # dedup
        ";

    let mut entries = Entries::parse(BASE, content.lines()).unwrap();

    let flat = entries.flatten(BASE, None).unwrap();

    assert!(entries.domains.is_none());

    assert_eq!(flat.dump().domain_keyword.len(), 1);
}
