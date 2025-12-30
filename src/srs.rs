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
    fn from(mut flat: FlatDomains) -> Self {
        let mut rule = Self {
            ..Default::default()
        };

        while let Some((kind, values)) = flat.take_next() {
            match kind {
                DomainKind::Suffix => rule.domain_suffix = Some(values),
                DomainKind::Full => rule.domain = Some(values),
                DomainKind::Keyword => rule.domain_keyword = Some(values),
                DomainKind::Regex => rule.domain_regex = Some(values),
            }
        }

        rule
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
            .unwrap()
            .as_ref(),
        "keyword"
    )
}
