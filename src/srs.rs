//! Adapter for [sing-box rule-set source format](https://sing-box.sagernet.org/configuration/rule-set/source-format)

use serde::Serialize;

use crate::FlatDomains;

/// [sing-box rule-set source structure](https://sing-box.sagernet.org/configuration/rule-set/source-format/#structure)
#[derive(Debug, Default, Serialize, Clone)]
pub struct RuleSet {
    pub version: u8,
    pub rules: Box<[Rule]>,
}

/// A single [sing-box headless rule](https://sing-box.sagernet.org/configuration/rule-set/headless-rule/#structure)
///
/// Note that the [`struct`][`Rule`] does not contain all
/// fields for sing-box headless rule
#[derive(Debug, Default, Serialize, Clone)]
#[serde_with::skip_serializing_none]
pub struct Rule {
    pub domain_suffix: Option<Box<[Box<str>]>>,
    pub domain: Option<Box<[Box<str>]>>,
    pub domain_keyword: Option<Box<[Box<str>]>>,
    pub domain_regex: Option<Box<[Box<str>]>>,
}

impl From<FlatDomains> for Rule {
    fn from(fd: FlatDomains) -> Self {
        let dump = fd.dump();
        Self {
            domain_suffix: Some(dump.domain_suffix).filter(|d| !d.is_empty()),
            domain: Some(dump.domain).filter(|d| !d.is_empty()),
            domain_keyword: Some(dump.domain_keyword).filter(|d| !d.is_empty()),
            domain_regex: Some(dump.domain_regex).filter(|d| !d.is_empty()),
        }
    }
}

impl FromIterator<Rule> for RuleSet {
    fn from_iter<T: IntoIterator<Item = Rule>>(iter: T) -> Self {
        Self {
            version: 1, // hard coded, no need for supporting extra rule fields
            rules: iter.into_iter().collect(),
        }
    }
}
