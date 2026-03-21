from serpapi import GoogleSearch
from company_verification_tool import SERP_API_KEY, CompanyVerificationTool

# from dotenv import load_dotenv
# import os
# from pathlib import Path

# # Go to project root (Group9-DS-and-AI-Lab-Project)
# BASE_DIR = Path(__file__).resolve().parents[3]

# env_path = BASE_DIR / ".env"

# print("Loading .env from:", env_path)

# load_dotenv(env_path)

# print("SERP_API_KEY:", os.getenv("SERP_API_KEY"))



# Test raw SerpAPI call
params = {
    "engine": "google",
    "q": "Infosys company",
    "api_key": SERP_API_KEY   # ✅ FIXED
}

search = GoogleSearch(params)
results = search.get_dict()

print("SERP API RESULT:")
print(results)

# Initialize tool
tool = CompanyVerificationTool(api_key=SERP_API_KEY)

# Test cases
test_cases = [
    {
        "company": "Infosys",
        "job_text": "Apply at https://www.infosys.com careers"
    },
    {
        "company": "XYZ Hiring Ltd",
        "job_text": "Apply now at http://xyz-careers-123.com and pay registration fee"
    },
    {
        "company": "Fake Job Corp",
        "job_text": "Send money to hr.fakejob@gmail.com"
    }
]

for test in test_cases:
    print("\n==============================")
    print(f"Testing: {test['company']}")
    
    result = tool.verify(
        company_name=test["company"],
        job_text=test["job_text"]
    )
    
    print(result)   # ✅ IMPORTANT