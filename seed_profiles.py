from memory.store import init_db, save_profile
from memory.schemas import UserProfile

init_db()

profiles = [
    UserProfile(
        user_id="pm_user",
        role="product_manager",
        team="growth",
        preferred_answer_style="concise_with_citations",
        preferred_doc_types=["PRD", "experiment_results"],
        recent_topics=["onboarding", "activation", "trial conversion"],
        frequent_docs=["prd_onboarding_v2"],
        session_context=["Q1 planning", "activation funnel"]
    ),
    UserProfile(
        user_id="eng_user",
        role="engineer",
        team="platform",
        preferred_answer_style="technical_depth",
        preferred_doc_types=["architecture", "reference"],
        recent_topics=["API", "infrastructure", "performance"],
        frequent_docs=["eng_arch_v1"],
        session_context=["backend scaling", "database optimization"]
    ),
    UserProfile(
        user_id="sales_user",
        role="sales_rep",
        team="sales",
        preferred_answer_style="bullet_points",
        preferred_doc_types=["battlecard"],
        recent_topics=["pricing", "competitors", "objections"],
        frequent_docs=["sales_battlecard_q1"],
        session_context=["enterprise deal", "competitive evaluation"]
    ),
]

for p in profiles:
    save_profile(p)
    print(f"Seeded: {p.user_id} ({p.role}, team={p.team})")

print("\nDone. 3 personas ready.")
