//! Adaptor for
//! [v2ray geosite](https://github.com/v2fly/v2ray-core/blob/master/app/router/routercommon/common.proto)
//!
//! This module contains some `impl From<A> for B`, refer to the source file
pub mod proto {
    //! Auto generated source file from `proto/geosite.proto` by [`prost`]
    include!(concat!(env!("OUT_DIR"), "/_.rs"));
}

use proto::{
    GeoSite, GeoSiteList,
    domain::{Attribute, Type, attribute::TypedValue},
};

use crate::{DomainKind, Entries, Entry, FlatDomains, Kind};

impl<S: AsRef<str>> From<S> for Attribute {
    fn from(value: S) -> Self {
        Self {
            key: value.as_ref().to_string(),
            typed_value: Some(TypedValue::BoolValue(true)),
        }
    }
}

impl From<Entry> for proto::Domain {
    fn from(entry: Entry) -> Self {
        let Kind::Domain(kind) = entry.kind else {
            unreachable!("not a domain kind");
        };
        let typ = match kind {
            DomainKind::Suffix => Type::RootDomain,
            DomainKind::Full => Type::Full,
            DomainKind::Keyword => Type::Plain,
            DomainKind::Regex => Type::Regex,
        };
        Self {
            r#type: typ.into(),
            value: entry.value.to_string(),
            attribute: entry.attrs.iter().map(Attribute::from).collect(),
        }
    }
}

impl From<FlatDomains> for GeoSite {
    fn from(flat: FlatDomains) -> Self {
        GeoSite {
            country_code: flat.base().to_owned(),
            domain: flat
                .into_vec()
                .into_iter()
                .map(proto::Domain::from)
                .collect(),
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
    fn from(entries: Entries) -> Self {
        let bases = entries.bases();

        Self {
            entry: bases
                .filter_map(|base| entries.flatten(&base, None))
                .map(GeoSite::from)
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sort_predictable() {
        crate::sort_predictable::helper(
            |domains| GeoSiteList::from_iter([GeoSite::from(domains)]),
            |list| {
                list.entry
                    .iter()
                    .map(|s| s.country_code.clone())
                    .collect::<Box<[_]>>()
            },
        );
    }

    fn generate_entries() -> Entries {
        let pairs = [
            ("base2", "regexp:regexp"),
            ("base1", "keyword:keyword"), // base (country_code) ord test
            ("base2", "regexp:regexp"),   // dedup
        ];
        let mut entries = crate::Entries::parse("base0", "full:full".lines());
        assert_eq!(
            pairs
                .iter()
                .map(|(base, content)| entries.parse_include(base, content.lines()))
                .count(),
            3
        );

        entries
    }

    #[test]
    fn from_iter() {
        let entries = generate_entries();

        let mut out = Vec::new();
        for base in entries.bases() {
            out.push(entries.flatten(&base, None).unwrap());
        }

        let geosite_list = GeoSiteList::from_iter(out.into_iter().map(GeoSite::from));

        assert_eq!(geosite_list.entry.len(), 3);

        let mut iter = geosite_list.entry.into_iter();
        assert_eq!(
            iter.next().unwrap(),
            GeoSite {
                country_code: "base0".to_owned(),
                domain: [proto::Domain {
                    r#type: Type::Full.into(),
                    value: "full".into(),
                    attribute: [].into()
                }]
                .into()
            }
        );
        assert_eq!(
            iter.next().unwrap(),
            GeoSite {
                country_code: "base1".to_owned(),
                domain: [proto::Domain {
                    r#type: Type::Plain.into(),
                    value: "keyword".into(),
                    attribute: [].into()
                }]
                .into()
            }
        );
        assert_eq!(
            iter.next().unwrap(),
            GeoSite {
                country_code: "base2".to_owned(),
                domain: [proto::Domain {
                    r#type: Type::Regex.into(),
                    value: "regexp".into(),
                    attribute: [].into()
                }]
                .into()
            }
        );
    }

    #[test]
    fn from_entries() {
        let entries = generate_entries();
        assert_eq!(
            GeoSiteList::from(generate_entries()),
            GeoSiteList::from_iter({
                entries
                    .bases()
                    .filter_map(|b| entries.flatten(&b, None))
                    .map(GeoSite::from)
            })
        );
    }
}
