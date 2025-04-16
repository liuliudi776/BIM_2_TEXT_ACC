# Automatic BIM Compliance Checking Based on Large Language Models

This repository supports automated rule checking for experiments and development related to the paper "BIM Automated Compliance Checking via the Large Language Models driven BIM-to-Text Paradigm."

---

## 1. Building Dataset (`building_data`)

- **rules/**
  Regulatory sentences in the building domain.

- **rule_types/**
  Predefined types of regulatory clauses in the building domain.

- **predefined_ner_tag/**
  Predefined entity types of regulatory sentences in the building domain.

- **ifc_model/**
  IFC model files in the building domain.

- **label/**
  Annotations for rule types, NER of regulatory entities, alignment between regulatory entities and IFC classes, manual compliance checking, and IFC entity alignment.

---

## 2. Bridge Dataset (`bridge_data`)

- **rules/**
  Regulatory sentences in the bridge domain.

- **rule_types/**
  Predefined types of regulatory clauses in the bridge domain.

- **predefined_ner_tag/**
  Predefined entity types of regulatory sentences in the bridge domain.

- **ifc_model/**
  IFC model files in the bridge domain.

- **label/**
  Annotations for rule types, NER of regulatory entities, alignment between regulatory entities and IFC classes, manual compliance checking, and IFC entity alignment.

---

## Automatic Compliance Checking (BIM ACC)
Steps for automatic compliance checking:
1. Run the command `pip install -r requirements.txt` to install the required project dependencies.
2. Configure the OpenAI model endpoint in `config/openai_api.toml`.
3. Edit `config/config.toml.example` under the `config` directory as needed, then save it as `config.toml`.
4. Run `python main.py --config config/config.toml` for compliance checking. Process information will be saved to `database/bridge_database` or `database/building_database`, and evaluation data will be saved to the `evaluate` directory.

---

## License Agreement
**The source code is freely available for research purposes to universities, research institutions, enterprises, and individuals, and must not be used for any commercial purpose.**