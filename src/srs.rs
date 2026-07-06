//! Adapter for [sing-box rule-set source format](https://sing-box.sagernet.org/configuration/rule-set/source-format)

use serde::Serialize;

use crate::{DomainKind, FlatDomains, Kind};

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
        let mut rule = Self::default();

        while let Some(domains) = flat.take_next() {
            let Kind::Domain(kind) = domains.first().unwrap().kind else {
                unreachable!("not a domain kind");
            };
            let mut values: Vec<_> = domains
                .into_iter()
                .map(|d| Box::from(d.value.as_ref()))
                .collect();
            // extra dedup, b.c. `FlatDomains` wont't dedup
            // when two entry has the same value and type,
            // but has different attrs (which are not used in `Rule`)
            values.dedup();
            let values = values.into_boxed_slice();
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
            // https://github.com/SagerNet/sing-box/blob/v1.12.14/route/rule/rule_set_local.go#L102
            // https://github.com/SagerNet/sing-box/blob/v1.12.14/route/rule/rule_set_remote.go#L168
            version: 1, // hard coded, no need for supporting extra rule fields
            rules,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BASE, Entries};

    #[test]
    fn sort_predictable() {
        crate::sort_predictable::helper(
            |domains| RuleSet::from_iter([Rule::from(domains)]),
            |list| list.rules.clone(),
        )
    }

    #[test]
    fn dedup_attr() {
        let mut entries = Entries::parse(
            BASE,
            "\
        domain:abc
        domain:abc @attr1
        "
            .lines(),
        );

        let rule = Rule::from(entries.flatten(BASE, None).unwrap());

        assert_eq!(rule.domain_suffix.unwrap().len(), 1)
    }

    #[test]
    fn from_iter() {
        const CONTENT: &str = "\
            keyword:keyword
            keyword:keyword # dedup
        ";

        let mut entries = Entries::parse(BASE, CONTENT.lines());
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
}
