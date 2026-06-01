"""
backend/db/seed_rag.py — Seeds ChromaDB with medical knowledge documents
Run once: python -m backend.db.seed_rag
"""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.memory.chroma_memory import add_documents, MEDICAL_COLLECTION, POLICY_COLLECTION, collection_count

MEDICAL_DOCS = [
    {
        "id": "med_001",
        "title": "Common Cold (Erkältung)",
        "text": (
            "The common cold (Rhinitis) is caused by over 200 different viruses, most commonly rhinovirus. "
            "Symptoms include runny nose, sore throat, cough, sneezing, headache, and mild fever (ICD-10: J00). "
            "Treatment is symptomatic: rest, fluids, nasal decongestants, and analgesics like paracetamol or ibuprofen. "
            "Antibiotics are NOT effective against viral infections. Most colds resolve in 7-10 days. "
            "See a doctor if symptoms worsen after 10 days, fever exceeds 39°C, or difficulty breathing occurs."
        ),
        "source": "https://www.rki.de",
        "language": "en",
    },
    {
        "id": "med_002",
        "title": "Fever Management",
        "text": (
            "Fever (Fieber, ICD-10: R50.9) is a body temperature above 38°C (100.4°F). "
            "Mild fever (38-39°C) often doesn't need medication — it helps fight infection. "
            "Use paracetamol (acetaminophen) or ibuprofen for fever above 38.5°C or discomfort. "
            "Seek emergency care if: fever above 40°C, febrile seizure, stiff neck, severe headache, "
            "confusion, difficulty breathing, or rash. In infants under 3 months, any fever requires "
            "immediate medical attention (call 116 117 or go to A&E)."
        ),
        "source": "https://www.bundesgesundheitsministerium.de",
        "language": "en",
    },
    {
        "id": "med_003",
        "title": "Chest Pain — When to Seek Help",
        "text": (
            "Chest pain (Brustschmerz, ICD-10: R07.9) has many causes. EMERGENCY causes: "
            "heart attack (myocardial infarction), pulmonary embolism, aortic dissection, tension pneumothorax. "
            "CALL 112 immediately if chest pain is: crushing/pressure-like, radiates to arm/jaw/back, "
            "accompanied by sweating/nausea/shortness of breath. Non-emergency causes include "
            "muscle strain, acid reflux (GERD), anxiety, and costochondritis. "
            "When in doubt, always call 112 — it's better to be safe."
        ),
        "source": "https://www.bundesgesundheitsministerium.de",
        "language": "en",
    },
    {
        "id": "med_004",
        "title": "High Blood Pressure (Hypertension)",
        "text": (
            "Hypertension (Bluthochdruck, ICD-10: I10) is blood pressure consistently above 140/90 mmHg. "
            "Risk factors include obesity, salt intake, smoking, alcohol, stress, family history, and age. "
            "Often symptomless ('silent killer'). Complications: stroke, heart attack, kidney disease, vision loss. "
            "Treatment includes lifestyle changes (DASH diet, exercise, reduced salt/alcohol) and medications "
            "(ACE inhibitors, beta-blockers, diuretics, calcium channel blockers). "
            "Regular monitoring essential. Hypertensive crisis (>180/120): call 112."
        ),
        "source": "https://www.rki.de",
        "language": "en",
    },
    {
        "id": "med_005",
        "title": "Diabetes Type 2 — Overview",
        "text": (
            "Type 2 Diabetes (ICD-10: E11) is characterised by insulin resistance and progressive beta-cell dysfunction. "
            "Symptoms: increased thirst/urination, fatigue, blurred vision, slow wound healing. "
            "Diagnosis: fasting glucose ≥126 mg/dL or HbA1c ≥6.5%. "
            "Treatment: lifestyle modification (diet, exercise, weight loss), Metformin as first-line medication, "
            "then GLP-1 agonists, SGLT-2 inhibitors, insulin if needed. "
            "Regular monitoring of glucose, HbA1c, kidney function, eyes (retinopathy), and feet essential. "
            "In Germany, specialised Diabetologen and Diabetesberater are available."
        ),
        "source": "https://www.rki.de",
        "language": "en",
    },
    {
        "id": "med_006",
        "title": "Mental Health — Depression and Anxiety",
        "text": (
            "Depression (ICD-10: F32) symptoms: persistent low mood, loss of interest, sleep/appetite changes, "
            "fatigue, concentration problems, guilt, and in severe cases suicidal thoughts. "
            "Anxiety disorders (ICD-10: F41) include GAD, panic disorder, and social anxiety. "
            "Treatment: psychotherapy (CBT, DBT), antidepressants (SSRIs, SNRIs), combination therapy. "
            "In Germany, Hausarzt can refer to a Psychiater or Psychotherapeut. "
            "Crisis support: Telefonseelsorge: 0800 111 0 111 (free, 24/7). "
            "For acute suicidal crisis: call 112 or go to nearest A&E."
        ),
        "source": "https://www.bundesgesundheitsministerium.de",
        "language": "en",
    },
    {
        "id": "med_007",
        "title": "COVID-19 — Current Guidance",
        "text": (
            "COVID-19 (SARS-CoV-2, ICD-10: U07.1) symptoms: fever, cough, fatigue, loss of taste/smell, "
            "shortness of breath, headache, muscle pain. Most cases are mild and self-limiting. "
            "High-risk groups: elderly, immunocompromised, chronic conditions. "
            "Testing: PCR and rapid antigen tests available at pharmacies. "
            "Treatment: rest, fluids, paracetamol for symptoms. Antivirals (Paxlovid) for high-risk patients. "
            "Seek medical attention if: difficulty breathing, persistent chest pain, confusion, "
            "oxygen saturation below 94%. Vaccination remains the best prevention."
        ),
        "source": "https://www.rki.de",
        "language": "en",
    },
    {
        "id": "med_008",
        "title": "Children's Health — Vaccinations in Germany",
        "text": (
            "Germany follows the STIKO (Ständige Impfkommission) vaccination schedule. "
            "Key vaccines: Rotavirus (6 weeks), Diphtheria/Tetanus/Pertussis/Polio/Hib/Hepatitis B (2,3,4 months), "
            "Pneumococcus (2,3,4 months), MMR/Varicella (11 months, 15 months), "
            "Meningococcus C (12 months), HPV (9-14 years). "
            "GKV covers all STIKO recommended vaccinations. "
            "U-Untersuchungen (preventive checkups) are scheduled from birth to age 6. "
            "Contact your Kinderarzt (paediatrician) for the vaccination booklet (Impfpass)."
        ),
        "source": "https://www.rki.de",
        "language": "en",
    },
]

POLICY_DOCS = [
    {
        "id": "pol_001",
        "title": "German Health Insurance System Overview",
        "text": (
            "Germany has two health insurance systems: "
            "GKV (Gesetzliche Krankenversicherung / Statutory) covering ~90% of population, and "
            "PKV (Private Krankenversicherung) for the remainder. "
            "GKV contributions: ~14.6% of gross salary (split employer/employee) plus supplementary rate (~1.7%). "
            "GKV covers: doctor visits, hospital treatment, medications (with co-pay), dental (basic), "
            "preventive care, mental health, physiotherapy, and more. "
            "Family members (spouse/children) can be co-insured for free. "
            "Major GKV providers in Oberhausen: AOK Rheinland, Barmer, TK, DAK-Gesundheit."
        ),
    },
    {
        "id": "pol_002",
        "title": "Healthcare Rights for Asylum Seekers (AsylbLG)",
        "text": (
            "Under §4 Asylbewerberleistungsgesetz (AsylbLG), asylum seekers receive a health treatment voucher "
            "(Krankenschein/Behandlungsschein) from the Sozialamt. "
            "First 18 months: covers acute illness, pain, childbirth, vaccinations (§4 AsylbLG). "
            "After 18 months (§6 AsylbLG): similar coverage to GKV. "
            "Emergency treatment is always available regardless of status. "
            "For chronic conditions, apply to Sozialamt for additional coverage. "
            "In NRW, contact the Sozialamt Oberhausen: Schwartzstr. 72, ☎ 0208 825-0."
        ),
    },
]


def seed_rag():
    """Seed ChromaDB with medical and policy documents."""
    print("🌱 Seeding RAG databases...")

    # Medical knowledge
    count_before = collection_count(MEDICAL_COLLECTION)
    if count_before == 0:
        docs = [d["text"] for d in MEDICAL_DOCS]
        metas = [{"id": d["id"], "title": d["title"], "source": d.get("source", ""), "language": d.get("language", "en")} for d in MEDICAL_DOCS]
        ids = [d["id"] for d in MEDICAL_DOCS]
        success = add_documents(MEDICAL_COLLECTION, docs, metas, ids)
        print(f"✅ Medical knowledge: added {len(docs)} documents" if success else "❌ Medical knowledge: failed")
    else:
        print(f"ℹ️  Medical knowledge: already has {count_before} documents")

    # Policy knowledge
    count_before = collection_count(POLICY_COLLECTION)
    if count_before == 0:
        docs = [d["text"] for d in POLICY_DOCS]
        metas = [{"id": d["id"], "title": d["title"]} for d in POLICY_DOCS]
        ids = [d["id"] for d in POLICY_DOCS]
        success = add_documents(POLICY_COLLECTION, docs, metas, ids)
        print(f"✅ Policy knowledge: added {len(docs)} documents" if success else "❌ Policy knowledge: failed")
    else:
        print(f"ℹ️  Policy knowledge: already has {count_before} documents")

    print("✅ RAG seeding complete")


if __name__ == "__main__":
    seed_rag()
