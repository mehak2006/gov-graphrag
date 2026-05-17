import wikipediaapi
import json
import os

wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='GovSchemesResearch/1.0'
)

# Large list of Indian govt scheme related articles
topics = [
    "Pradhan Mantri Jan Dhan Yojana",
    "Pradhan Mantri Awas Yojana",
    "Mahatma Gandhi National Rural Employment Guarantee Act",
    "Ayushman Bharat",
    "Pradhan Mantri Kisan Samman Nidhi",
    "Swachh Bharat Mission",
    "Digital India",
    "Make in India",
    "Startup India",
    "Skill India",
    "Pradhan Mantri Mudra Yojana",
    "Pradhan Mantri Ujjwala Yojana",
    "National Health Mission",
    "Mid-Day Meal Scheme",
    "Integrated Child Development Services",
    "National Social Assistance Programme",
    "Rashtriya Krishi Vikas Yojana",
    "Pradhan Mantri Fasal Bima Yojana",
    "Soil Health Card",
    "Paramparagat Krishi Vikas Yojana",
    "National Food Security Act",
    "Public Distribution System",
    "Pradhan Mantri Gram Sadak Yojana",
    "Pradhan Mantri Garib Kalyan Yojana",
    "Atal Pension Yojana",
    "Pradhan Mantri Jeevan Jyoti Bima Yojana",
    "Pradhan Mantri Suraksha Bima Yojana",
    "National Pension System",
    "Employees Provident Fund",
    "Employees State Insurance",
    "Pradhan Mantri Matru Vandana Yojana",
    "Beti Bachao Beti Padhao",
    "Sukanya Samriddhi Yojana",
    "National Rural Health Mission",
    "Janani Suraksha Yojana",
    "Rashtriya Swasthya Bima Yojana",
    "Pradhan Mantri Swasthya Suraksha Yojana",
    "National AIDS Control Programme",
    "Pulse Polio Programme",
    "Mission Indradhanush",
    "Sarva Shiksha Abhiyan",
    "Rashtriya Madhyamik Shiksha Abhiyan",
    "National Literacy Mission",
    "Pradhan Mantri Kaushal Vikas Yojana",
    "National Apprenticeship Promotion Scheme",
    "Deen Dayal Upadhyaya Grameen Kaushalya Yojana",
    "National Urban Livelihood Mission",
    "National Rural Livelihood Mission",
    "MNREGA",
    "Pradhan Mantri Rozgar Protsahan Yojana",
    "Stand Up India",
    "Pradhan Mantri Employment Generation Programme",
    "Khadi and Village Industries Commission",
    "National Handicapped Finance and Development Corporation",
    "National Minorities Development and Finance Corporation",
    "National Scheduled Castes Finance and Development Corporation",
    "National Scheduled Tribes Finance and Development Corporation",
    "National Backward Classes Finance and Development Corporation",
    "Post Matric Scholarship",
    "National Fellowship",
    "Prime Minister Research Fellowship",
    "National Talent Search Examination",
    "Inspire Scholarship",
    "Central Sector Scheme of Scholarship",
    "National Means cum Merit Scholarship",
    "Pragati Scholarship",
    "Saksham Scholarship",
    "Ishan Uday",
    "National Overseas Scholarship",
    "Rajiv Gandhi National Fellowship",
    "Maulana Azad National Fellowship",
    "Indira Gandhi Scholarship",
    "Single Girl Child Scholarship",
    "Top Class Education Scheme",
    "Free Coaching Scheme",
    "Pre Matric Scholarship",
    "National Programme for Education of Girls",
    "Kasturba Gandhi Balika Vidyalaya",
    "Pradhan Mantri Vidya Lakshmi Karyakram",
    "Deen Dayal Upadhyaya Antyodaya Yojana",
    "Pradhan Mantri Avas Yojana Urban",
    "Housing for All",
    "Smart Cities Mission",
    "Atal Mission for Rejuvenation",
    "AMRUT",
    "Pradhan Mantri Sahaj Bijli Har Ghar Yojana",
    "Ujwal DISCOM Assurance Yojana",
    "National Solar Mission",
    "Pradhan Mantri Krishi Sinchai Yojana",
    "Har Ghar Jal",
    "Swajal",
    "Namami Gange",
    "National Ganga River Basin Authority",
    "Green India Mission",
    "National Mission for a Green India",
    "Soil and Water Conservation",
    "Pradhan Mantri Van Dhan Yojana",
    "Eklavya Model Residential Schools",
    "Tribal Sub Plan",
    "Special Central Assistance to Tribal Sub Plan",
    "Forest Rights Act",
    "Scheduled Castes Sub Plan",
    "Special Component Plan",
]

documents = []
total_chars = 0
target = 5_000_000  # 5 million chars ~ 1.25M tokens

print(f"Downloading {len(topics)} articles...")

for i, topic in enumerate(topics):
    try:
        page = wiki.page(topic)
        if page.exists():
            text = page.text
            documents.append({
                "title": topic,
                "content": text,
                "source": f"wikipedia:{topic}"
            })
            total_chars += len(text)
            print(f"[{i+1}/{len(topics)}] {topic}: {len(text)} chars | Total: {total_chars:,}")
        else:
            print(f"[{i+1}/{len(topics)}] NOT FOUND: {topic}")
    except Exception as e:
        print(f"Error: {topic} - {e}")

# Save as GraphRAG JSON
output = {"documents": documents}
os.makedirs("/home/mehak2006/gov-graphrag/data", exist_ok=True)
with open("/home/mehak2006/gov-graphrag/data/govt_schemes_large.json", "w") as f:
    json.dump(output, f, indent=2)

tokens_estimate = total_chars // 4
print(f"\nDone! {len(documents)} articles, ~{tokens_estimate:,} tokens")
print(f"Saved to: govt_schemes_large.json")
