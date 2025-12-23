//! Adaptor for
//! [v2ray geosite](https://github.com/v2fly/v2ray-core/blob/master/app/router/routercommon/common.proto)
//!
//! This module contains some `impl From<A> for B`, refer to the source file
use std::rc::Rc;

pub mod proto {
    //! Auto generated source file from `proto/geosite.proto` by [`prost`]
    include!(concat!(env!("OUT_DIR"), "/_.rs"));
}

use proto::{
    domain::{attribute::TypedValue, Attribute, Type},
    GeoSite, GeoSiteList,
};

use crate::{DomainKind, FlatDomains};

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
    fn from(fd: FlatDomains) -> Self {
        let domains = fd.0;
        GeoSite {
            country_code: domains[0].base.clone().to_string(), // it's flattened
            domain: domains.into_iter().map(proto::Domain::from).collect(),
        }
    }
}

impl FromIterator<FlatDomains> for GeoSiteList {
    fn from_iter<T: IntoIterator<Item = FlatDomains>>(iter: T) -> Self {
        Self {
            entry: iter.into_iter().map(GeoSite::from).collect(),
        }
    }
}
