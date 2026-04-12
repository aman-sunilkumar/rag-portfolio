import yaml
from openai import OpenAI
from dotenv import load_dotenv
from memory.store import get_recent_queries
from memory.schemas import UserProfile

load_dotenv()

with open("prompts/config.yaml") as f:
    config = yaml.safe_load(f)

client = OpenAI()


def expand_query(query: str, profile: UserProfile, recent_queries: list) -> str:
    """Rewrite the query using user context to improve retrieval precision."""
    if not recent_queries and not profile.recent_topics:
        return query  # cold start — return raw query unchanged

    context_parts = []
    if recent_queries:
        context_parts.append(f"Recent questions: {'; '.join(recent_queries[-3:])}")
    if profile.recent_topics:
        context_parts.append(f"User's recurring interests: {', '.join(profile.recent_topics)}")
    if profile.session_context:
        context_parts.append(f"Current session context: {', '.join(profile.session_context)}")

    prompt = f"""You are a search query optimizer.

User role: {profile.role}, team: {profile.team}
{chr(10).join(context_parts)}

Original query: "{query}"

Rewrite the query to improve document retrieval given this user's context.
Return ONLY the rewritten query. Keep it under 25 words. No quotes."""

    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        max_tokens=60,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content.strip()
