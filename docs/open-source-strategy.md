# Open-Source Strategy (MVP)

## Licensing

- Current repository license: MIT.
- Planned production release: permissive license for core decision-support components, with clear module boundaries for any deployment-specific integrations.

## What is open source

The following components are intended for public open-source release:

- feature ingestion adapters for non-sensitive climate and aggregated health indicators,
- risk scoring pipeline and transparent weighting logic,
- alert generation templates and orchestration logic,
- deployment and implementation documentation.

## Data privacy by design

- No personal health information is included in this repository.
- Reference datasets are synthetic/anonymized.
- Deployment guidance requires local legal and ethical review before integrating any sensitive data source.

## Collaboration model

- Public issue tracker for bugs/features.
- Contribution guide and code of conduct in later iterations.
- Country adaptation playbooks to support localization across UNICEF programme contexts.
