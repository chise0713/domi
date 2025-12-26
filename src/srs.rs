//! Adapter for [sing-box rule-set source format](https://sing-box.sagernet.org/configuration/rule-set/source-format)

use serde::Serialize;

use crate::{DomainKind, FlatDomains};

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
#[serde_with::skip_serializing_none]
#[derive(Debug, Default, Serialize, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Rule {
    pub domain_suffix: Option<Box<[Box<str>]>>,
    pub domain: Option<Box<[Box<str>]>>,
    pub domain_keyword: Option<Box<[Box<str>]>>,
    pub domain_regex: Option<Box<[Box<str>]>>,
}

impl From<FlatDomains> for Rule {
    fn from(flat: FlatDomains) -> Self {
        let mut domains = flat.into_vec();
        let mut take = |kind: DomainKind| -> Option<Box<[Box<str>]>> {
            let idx = domains.partition_point(|d| d.kind != kind);
            let v: Box<[_]> = domains
                .split_off(idx)
                .into_iter()
                .map(|d| d.value)
                .collect();
            (!v.is_empty()).then_some(v)
        };

        Self {
            domain_suffix: take(DomainKind::Suffix),
            domain: take(DomainKind::Full),
            domain_keyword: take(DomainKind::Keyword),
            domain_regex: take(DomainKind::Regex),
        }
    }
}

impl FromIterator<Rule> for RuleSet {
    fn from_iter<T: IntoIterator<Item = Rule>>(iter: T) -> Self {
        // for output consistency
        let mut rules: Box<_> = iter.into_iter().collect();
        rules.sort();
        Self {
            version: 1, // hard coded, no need for supporting extra rule fields
            rules,
        }
    }
}

#[test]
fn test_sort_predictable() {
    crate::sort_predictable::test(
        |domains| RuleSet::from_iter([Rule::from(domains)]),
        |list| list.rules.clone(),
    )
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
    let rule_set = RuleSet::from_iter([Rule::from(flattened)]);
    assert_eq!(rule_set.rules.len(), 1);
    assert_eq!(
        rule_set
            .rules
            .into_iter()
            .next()
            .unwrap()
            .domain_keyword
            .unwrap()
            .into_iter()
            .next()
            .unwrap(),
        "keyword".into()
    )
}
