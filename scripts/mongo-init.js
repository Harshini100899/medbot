// MongoDB initialization for MedBot
db = db.getSiblingDB('medbot');

// Create collections with validators
db.createCollection('sessions');
db.createCollection('conversations');
db.createCollection('doctors');
db.createCollection('pharmacies');
db.createCollection('medical_ontology');

// ─── Indexes ──────────────────────────────────────────────────────────────────
db.sessions.createIndex({ "session_id": 1 }, { unique: true });
db.sessions.createIndex({ "created_at": 1 }, { expireAfterSeconds: 86400 * 7 }); // 7 days TTL

db.conversations.createIndex({ "session_id": 1 });
db.conversations.createIndex({ "timestamp": -1 });
db.conversations.createIndex({ "language": 1 });

db.doctors.createIndex({ "name": "text", "specialization": "text", "city": "text" });
db.doctors.createIndex({ "location": "2dsphere" });
db.doctors.createIndex({ "specialization": 1 });
db.doctors.createIndex({ "city": 1 });

db.pharmacies.createIndex({ "location": "2dsphere" });
db.pharmacies.createIndex({ "city": 1 });

// ─── Seed Doctor Data (Oberhausen region) ─────────────────────────────────────
db.doctors.insertMany([
  {
    name: "Dr. Maria Schmidt",
    specialization: "General Practitioner",
    address: "Marktstraße 45, 46045 Oberhausen",
    city: "Oberhausen",
    phone: "+49 208 123456",
    languages: ["de", "en"],
    kvno_accepted: true,
    location: { type: "Point", coordinates: [6.8630, 51.4696] },
    available: true
  },
  {
    name: "Dr. Ahmet Yilmaz",
    specialization: "Internal Medicine",
    address: "Mülheimer Str. 12, 46045 Oberhausen",
    city: "Oberhausen",
    phone: "+49 208 654321",
    languages: ["de", "tr", "en"],
    kvno_accepted: true,
    location: { type: "Point", coordinates: [6.8740, 51.4720] },
    available: true
  },
  {
    name: "Dr. Olena Kovalenko",
    specialization: "Pediatrics",
    address: "Bahnhofstr. 8, 46045 Oberhausen",
    city: "Oberhausen",
    phone: "+49 208 789012",
    languages: ["de", "uk", "en"],
    kvno_accepted: true,
    location: { type: "Point", coordinates: [6.8580, 51.4680] },
    available: true
  },
  {
    name: "Dr. Fatma Demir",
    specialization: "Gynecology",
    address: "Königstr. 22, 46045 Oberhausen",
    city: "Oberhausen",
    phone: "+49 208 345678",
    languages: ["de", "tr"],
    kvno_accepted: true,
    location: { type: "Point", coordinates: [6.8650, 51.4710] },
    available: true
  },
  {
    name: "Dr. Klaus Weber",
    specialization: "Orthopedics",
    address: "Elsässer Str. 30, 46117 Oberhausen",
    city: "Oberhausen",
    phone: "+49 208 901234",
    languages: ["de", "en"],
    kvno_accepted: true,
    location: { type: "Point", coordinates: [6.8710, 51.4750] },
    available: true
  },
  {
    name: "Evangelisches Krankenhaus Oberhausen",
    specialization: "Hospital - Emergency",
    address: "Virchowstr. 20, 46047 Oberhausen",
    city: "Oberhausen",
    phone: "+49 208 881-0",
    languages: ["de", "en", "tr"],
    kvno_accepted: true,
    location: { type: "Point", coordinates: [6.8541, 51.4767] },
    available: true,
    is_hospital: true
  },
  {
    name: "St. Marien-Hospital Oberhausen",
    specialization: "Hospital - Emergency",
    address: "Josefstr. 3, 46045 Oberhausen",
    city: "Oberhausen",
    phone: "+49 208 8999-0",
    languages: ["de", "en"],
    kvno_accepted: true,
    location: { type: "Point", coordinates: [6.8600, 51.4700] },
    available: true,
    is_hospital: true
  }
]);

// ─── Seed Pharmacy Data ────────────────────────────────────────────────────────
db.pharmacies.insertMany([
  {
    name: "Rathaus-Apotheke Oberhausen",
    address: "Marktstraße 1, 46045 Oberhausen",
    city: "Oberhausen",
    phone: "+49 208 200100",
    hours: "Mo-Fr 08:00-18:30, Sa 09:00-14:00",
    night_service: false,
    location: { type: "Point", coordinates: [6.8620, 51.4695] }
  },
  {
    name: "Glückauf-Apotheke",
    address: "Mülheimer Str. 50, 46045 Oberhausen",
    city: "Oberhausen",
    phone: "+49 208 855000",
    hours: "Mo-Fr 08:00-20:00, Sa 09:00-16:00",
    night_service: true,
    location: { type: "Point", coordinates: [6.8730, 51.4725] }
  }
]);

print("MedBot MongoDB initialized successfully.");
