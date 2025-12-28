import os
import sys
import asyncio

sys.path.insert(0, os.getcwd())

from backend.travel_agent.tools import send_email_notification, SENT_EMAILS


async def _run():
    # ensure clean state
    SENT_EMAILS.clear()

    to = "user@example.com"
    subject = "Test Subject"
    body = "This is a test email body."

    # Force mock email mode for test regardless of environment
    try:
        import backend.travel_agent.config as _cfg
        import backend.travel_agent.tools as _tools
        # force mock mode by clearing tools module-level credentials (tools imported them at module load)
        _tools.EMAIL_SENDER = None
        _tools.EMAIL_PASSWORD = None
    except Exception:
        pass

    r1 = await send_email_notification.ainvoke({"to_email": to, "subject": subject, "body": body})
    r2 = await send_email_notification.ainvoke({"to_email": to, "subject": subject, "body": body})

    print("first:", r1)
    print("second:", r2)

    assert "mock" in str(r1).lower() or "sent" in str(r1).lower()
    assert str(r2) == "Skipped duplicate email (idempotent)."


if __name__ == '__main__':
    import asyncio

    asyncio.run(_run())
    print('OK')
