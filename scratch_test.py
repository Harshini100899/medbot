"""Quick smoke test to verify the pipeline works end-to-end."""
import asyncio
import sys
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

async def main():
    # Test 1: Intent routing
    from backend.agents.supervisor_agent import rule_based_intent
    tests = [
        ("i have a diabetes find a suitable doctor in oberhausen", "doctor_search"),
        ("i have heart problems find doctor", "doctor_search"),
        ("find me a specialist in oberhausen", "doctor_search"),
        ("what are symptoms of flu", None),  # will go to LLM
        ("call 112 heart attack", "emergency"),
    ]
    print("=== Intent Routing Tests ===")
    all_pass = True
    for msg, expected in tests:
        got = rule_based_intent(msg)
        status = "PASS" if (got == expected or (expected is None and got != "doctor_search")) else "FAIL"
        print(f"[{status}] '{msg[:55]}' => {got} (expected {expected})")
        if status == "FAIL":
            all_pass = False

    # Test 2: Direct scraper
    print("\n=== Direct Scraper Test ===")
    from backend.tools.arzt_auskunft_scraper import scrape_arzt_auskunft
    try:
        docs = await scrape_arzt_auskunft("diabetes", city="Oberhausen", limit=3)
        if docs:
            print(f"[PASS] Scraped {len(docs)} doctors from arzt-auskunft.de:")
            for d in docs:
                name = d['name'].encode('ascii', 'replace').decode('ascii')
                spec = d['specialization'].encode('ascii', 'replace').decode('ascii')
                addr = d['address'].encode('ascii', 'replace').decode('ascii')
                print(f"   * {name} | {spec}")
                print(f"     {addr} | {d['source_url']}")
        else:
            print("[WARN] Scraper returned empty (might be network issue)")
    except Exception as e:
        print(f"[FAIL] Scraper error: {e}")

    # Test 3: Full find_doctors pipeline
    print("\n=== find_doctors Pipeline Test ===")
    from backend.tools.doctor_search_tool import find_doctors
    result = await find_doctors("i have a diabetes find a suitable doctor", city="Oberhausen")
    docs = result.get("doctors", [])
    spec = result.get("inferred_specialisation")
    print(f"Inferred specialisation: {spec}")
    print(f"Doctors found: {len(docs)}")
    for d in docs[:3]:
        name = d['name'].encode('ascii', 'replace').decode('ascii')
        sp = d['specialization'].encode('ascii', 'replace').decode('ascii')
        addr = d['address'].encode('ascii', 'replace').decode('ascii')
        print(f"   * {name} | {sp} | {addr}")

    if all_pass and docs:
        print("\n[ALL PASS] All smoke tests passed!")
    else:
        print("\n[WARN] Some issues above - check output.")

asyncio.run(main())
