//! Adaptor for
//! [v2ray geosite](https://github.com/v2fly/v2ray-core/blob/master/app/router/routercommon/common.proto)
//!
//! This module contains some `impl From<A> for B`, refer to the source file
pub mod proto {
    //! Auto generated source file from `proto/geosite.proto` by [`prost`]
    include!(concat!(env!("OUT_DIR"), "/_.rs"));
}

use std::rc::Rc;

use proto::{
    domain::{attribute::TypedValue, Attribute, Type},
    GeoSite, GeoSiteList,
};

use crate::{DomainKind, Entries, FlatDomains};

impl From<Rc<str>> for Attribute {
    fn from(value: Rc<str>) -> Self {
        Self {
            key: value.to_string(),
            typed_value: Some(TypedValue::BoolValue(true)),
        }
    }
}

impl From<crate::Domain> for proto::Domain {
    fn from(crate_domain: crate::Domain) -> Self {
        let typ = match crate_domain.kind {
            DomainKind::Suffix => Type::RootDomain,
            DomainKind::Full => Type::Full,
            DomainKind::Keyword => Type::Plain,
            DomainKind::Regex => Type::Regex,
        };
        Self {
            r#type: typ.into(),
            value: crate_domain.value.to_string(),
            attribute: crate_domain
                .attrs
                .into_iter()
                .map(Attribute::from)
                .collect(),
        }
    }
}

impl From<FlatDomains> for GeoSite {
    fn from(flat: FlatDomains) -> Self {
        let domains = flat.into_vec();
        GeoSite {
            country_code: domains[0].base.to_string(), // it's flattened
            domain: domains.into_iter().map(proto::Domain::from).collect(),
        }
    }
}

impl FromIterator<GeoSite> for GeoSiteList {
    fn from_iter<T: IntoIterator<Item = GeoSite>>(iter: T) -> Self {
        // for output consistency
        let mut entry: Vec<_> = iter.into_iter().collect();
        entry.sort_by(|a, b| a.country_code.cmp(&b.country_code));
        Self { entry }
    }
}

impl From<Entries> for GeoSiteList {
    fn from(mut entries: Entries) -> Self {
        let bases = entries.bases();

        GeoSiteList::from_iter(
            bases
                .filter_map(|base| entries.flatten(&base, None))
                .map(GeoSite::from),
        )
    }
}

#[test]
fn test_sort_predictable() {
    crate::sort_predictable::test(
        |domains| GeoSiteList::from_iter([GeoSite::from(domains)]),
        |list| {
            list.entry
                .iter()
                .map(|s| s.country_code.clone())
                .collect::<Box<[_]>>()
        },
    );
}

#[test]
fn test_from_iter() {
    use crate::BASE;
    const CONTENT: &str = "\
            keyword:keyword
            keyword:keyword # dedup
        ";

    let mut entries = crate::Entries::parse(BASE, CONTENT.lines());
    let flattened = entries.flatten(BASE, None).unwrap();
    let geosite_list = GeoSiteList::from_iter([GeoSite::from(flattened)]);
    assert_eq!(geosite_list.entry.len(), 1);
    assert_eq!(
        geosite_list.entry.into_iter().next().unwrap(),
        GeoSite {
            country_code: BASE.into(),
            domain: [proto::Domain {
                r#type: Type::Plain.into(),
                value: "keyword".into(),
                attribute: [].into()
            }]
            .into()
        }
    )
}

#[test]
fn test_from_entries() {
    let pairs = [
        ("base1", "keyword:keyword"),
        ("base2", "regexp:regexp"),
        ("base2", "regexp:regexp"),
    ]; // dedup
    let mut entries = crate::Entries::parse("base0", "full:full".lines());
    assert_eq!(
        pairs
            .iter()
            .map(|(base, content)| entries.parse_extend(base, content.lines()))
            .count(),
        3
    );
    let geosite_list = GeoSiteList::from(entries);
    assert_eq!(
        geosite_list.entry.len(),
        3 // base0 + base1 + base2
    )
}
