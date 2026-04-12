import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from memory.store import init_db, save_profile
from memory.schemas import UserProfile
from personalized_rag import load_vectorstore, load_retriever, personalized_query

QUESTION = "What are the key risks and how should we handle them?"

PERSONAS = [
    UserProfile(user_id="eval_pm", role="product_manager", team="growth",
                preferred_doc_types=["PRD"], recent_topics=["onboarding","activation"],
                session_context=["Q1 planning"]),
    UserProfile(user_id="eval_eng", role="engineer", team="platform",
                preferred_doc_types=["architecture"], recent_topics=["API","infrastructure"],
                session_context=["backend scaling"]),
    UserProfile(user_id="eval_sales", role="sales_rep", team="sales",
                preferred_doc_types=["battlecard"], recent_topics=["pricing","objections"],
                session_context=["enterprise deal"]),
]

def run():
    init_db()
    for p in PERSONAS:
        save_profile(p)

    vs = load_vectorstore()
    retriever = load_retriever(vs)

    print(f"\nQuestion: {QUESTION}\n{'='*55}")
    for persona in PERSONAS:
        result = personalized_query(QUESTION, persona.user_id, retriever)
        sources = [(s["source"], s["signals"]["role_match"]) for s in result["sources"]]
        print(f"\n[{persona.role}]")
        print(f"  Expanded: {result['expanded_query']}")
        print(f"  Sources retrieved:")
        for src, rm in sources:
            print(f"    {src}  (role_match={rm})")
        print(f"  Answer preview: {result['answer'][:150]}...")

    print("\n" + "="*55)
    print("If pm_user pulled PRD, eng_user pulled arch doc,")
    print("and sales_user pulled battlecard — personalization is working.")

if __name__ == "__main__":
    run()
