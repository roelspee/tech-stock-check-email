"""
Tech Stock Price Tracker — AI Agent (Railway Cron Job)
=======================================================
Runs once per invocation — triggered by Railway's Cron Job service.
Recommended schedule: 0 7 * * *  (8am CET / UTC+1)

On each run, for every stock in WATCHLIST:
  1. Fetches current price
  2. If below alert threshold: fetches news, runs Claude analysis, sends email
  3. Exits

Environment variables (set in Railway):
    EMAIL_SENDER, EMAIL_RECEIVER
    SENDGRID_API_KEY
    ANTHROPIC_API_KEY
    NEWS_API_KEY
"""

import yfinance as yf
import os
import sys
import requests
import anthropic
from datetime import datetime
from typing import Optional


# ─────────────────────────────────────────────
#  WATCHLIST  — prices & thresholds as of March 5, 2026
#  alert_below = -5% of price at time of setup
#  Update these periodically to track current levels
# ─────────────────────────────────────────────

WATCHLIST = [
    {"ticker": "META",  "name": "Meta Platforms",    "alert_below": 634.58, "news_query": "META Facebook stock"},
    {"ticker": "GOOGL", "name": "Alphabet (Google)",  "alert_below": 287.97, "news_query": "Alphabet Google stock"},
    {"ticker": "MSFT",  "name": "Microsoft",          "alert_below": 384.94, "news_query": "Microsoft MSFT stock"},
    {"ticker": "AAPL",  "name": "Apple",              "alert_below": 249.39, "news_query": "Apple AAPL stock"},
    {"ticker": "NVDA",  "name": "Nvidia",             "alert_below": 173.89, "news_query": "Nvidia NVDA stock"},
    {"ticker": "AMZN",  "name": "Amazon",             "alert_below": 205.98, "news_query": "Amazon AMZN stock"},
]

LOG_FILE = "stock_price_log.csv"   # Set to None to disable

# --- Credentials (Railway environment variables) ---
EMAIL_SENDER      = os.environ.get("EMAIL_SENDER",      "you@gmail.com")
EMAIL_RECEIVER    = os.environ.get("EMAIL_RECEIVER",    "you@gmail.com")
SENDGRID_API_KEY  = os.environ.get("SENDGRID_API_KEY",  "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
NEWS_API_KEY      = os.environ.get("NEWS_API_KEY",      "")

# ─────────────────────────────────────────────


def get_price(ticker: str) -> Optional[float]:
    """Fetch the latest stock price."""
    try:
        stock = yf.Ticker(ticker)
        price = stock.fast_info.last_price
        return round(price, 2)
    except Exception as e:
        print(f"  [ERROR] Could not fetch price for {ticker}: {e}")
        return None


def get_news(query: str, num_articles: int = 5) -> list:
    """Fetch recent news headlines using NewsAPI."""
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "sortBy": "publishedAt",
            "pageSize": num_articles,
            "language": "en",
            "apiKey": NEWS_API_KEY,
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if data.get("status") != "ok":
            print(f"  [WARN] NewsAPI error: {data.get('message')}")
            return []

        return [
            {
                "title":       a.get("title", ""),
                "description": a.get("description", ""),
                "source":      a.get("source", {}).get("name", ""),
                "publishedAt": a.get("publishedAt", "")[:10],
                "url":         a.get("url", ""),
            }
            for a in data.get("articles", [])
        ]

    except Exception as e:
        print(f"  [ERROR] Could not fetch news: {e}")
        return []


def analyze_with_claude(ticker: str, name: str, price: float, target: float, articles: list) -> str:
    """Ask Claude to analyze the price drop in context of recent news."""
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        news_text = "\n".join([
            f"- [{a['source']}] {a['title']}: {a['description']}"
            for a in articles
        ]) if articles else "No recent news articles found."

        prompt = f"""You are a financial analyst assistant. {name} ({ticker}) stock has dropped below a user's alert threshold.

Price data:
- Current price: ${price:.2f}
- Alert threshold: ${target:.2f}
- Drop: ${target - price:.2f} ({((target - price) / target * 100):.2f}%)

Recent {name} news:
{news_text}

Provide a brief analysis covering:
1. Likely cause of the drop based on the news
2. Short-term dip or something more significant?
3. What to watch next

Write in plain prose paragraphs only. Do NOT use markdown formatting — no ## headers, no **bold**, no bullet points, no *italics*. Just clean sentences.
Be direct and practical. No explicit buy/sell advice."""

        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text

    except Exception as e:
        print(f"  [ERROR] Claude analysis failed: {e}")
        return "AI analysis unavailable at this time."


def markdown_to_html(text: str) -> str:
    """Convert markdown-flavoured text to safe inline HTML for emails."""
    import re
    lines = text.split("\n")
    html_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Strip ## / ### headings → bold paragraph
        line = re.sub(r'^#{1,3}\s+', '', line)
        # Remove leading bullet chars
        line = re.sub(r'^[-*]\s+', '', line)
        # **bold** → <strong>
        line = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line)
        # *italic* → <em>
        line = re.sub(r'\*(.+?)\*', r'<em>\1</em>', line)
        html_lines.append(f"<p style='margin:0 0 10px 0;'>{line}</p>")
    return "\n".join(html_lines)


def build_html_email(ticker: str, name: str, price: float, target: float, analysis: str, articles: list) -> tuple[str, str]:
    """Build subject line and HTML body for the alert email."""
    timestamp = datetime.now().strftime("%b %d, %Y · %H:%M")
    drop_usd  = target - price
    drop_pct  = (drop_usd / target) * 100

    news_rows = ""
    for a in articles[:4]:
        news_rows += f"""
        <tr>
          <td style="padding:10px 0; border-bottom:1px solid #f0f0f0;">
            <a href="{a['url']}" style="color:#1877f2; text-decoration:none; font-weight:600; font-size:14px;">{a['title']}</a>
            <div style="color:#888; font-size:12px; margin-top:3px;">{a['source']} &nbsp;·&nbsp; {a['publishedAt']}</div>
          </td>
        </tr>"""

    if not news_rows:
        news_rows = "<tr><td style='padding:10px 0; color:#888; font-size:14px;'>No recent news found.</td></tr>"

    analysis_html = markdown_to_html(analysis)

    subject = f"🔴 {ticker} ${price:.2f} — dropped ${drop_usd:.2f} below your alert"

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="margin:0; padding:0; background:#f4f5f7; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#f4f5f7; padding:32px 0;">
    <tr>
      <td align="center">
        <table width="560" cellpadding="0" cellspacing="0" style="background:#fff; border-radius:12px; overflow:hidden; box-shadow:0 2px 8px rgba(0,0,0,0.08);">

          <!-- Header -->
          <tr>
            <td style="background:#1877f2; padding:24px 32px;">
              <div style="color:#fff; font-size:11px; font-weight:700; letter-spacing:1.2px; text-transform:uppercase; opacity:0.8;">Stock Alert · {ticker}</div>
              <div style="color:#fff; font-size:26px; font-weight:700; margin-top:4px;">{name}</div>
              <div style="color:rgba(255,255,255,0.75); font-size:13px; margin-top:2px;">{timestamp}</div>
            </td>
          </tr>

          <!-- Price Block -->
          <tr>
            <td style="padding:28px 32px 0;">
              <table width="100%" cellpadding="0" cellspacing="0">
                <tr>
                  <td style="background:#fff5f5; border:1px solid #ffd0d0; border-radius:8px; padding:16px 20px;">
                    <div style="color:#c0392b; font-size:11px; font-weight:700; letter-spacing:1px; text-transform:uppercase;">Price dropped below target</div>
                    <div style="margin-top:8px;">
                      <span style="font-size:36px; font-weight:800; color:#1a1a1a;">${price:.2f}</span>
                      <span style="color:#c0392b; font-size:15px; font-weight:600; margin-left:10px;">▼ ${drop_usd:.2f} ({drop_pct:.1f}%)</span>
                    </div>
                    <div style="color:#888; font-size:13px; margin-top:4px;">Your alert threshold: <strong>${target:.2f}</strong></div>
                  </td>
                </tr>
              </table>
            </td>
          </tr>

          <!-- AI Analysis -->
          <tr>
            <td style="padding:24px 32px 0;">
              <div style="font-size:11px; font-weight:700; letter-spacing:1px; text-transform:uppercase; color:#888; margin-bottom:10px;">AI Analysis</div>
              <div style="font-size:14px; line-height:1.7; color:#333;">{analysis_html}</div>
            </td>
          </tr>

          <!-- News -->
          <tr>
            <td style="padding:24px 32px 0;">
              <div style="font-size:11px; font-weight:700; letter-spacing:1px; text-transform:uppercase; color:#888; margin-bottom:6px;">Recent News</div>
              <table width="100%" cellpadding="0" cellspacing="0">{news_rows}</table>
            </td>
          </tr>

          <!-- Footer -->
          <tr>
            <td style="padding:24px 32px 32px; color:#aaa; font-size:12px; border-top:1px solid #f0f0f0; margin-top:24px;">
              Automated alert · Runs daily at 08:00 CET via Railway Cron
            </td>
          </tr>

        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""

    return subject, html


def send_smart_email(ticker: str, name: str, price: float, target: float, analysis: str, articles: list) -> bool:
    """Send HTML alert email via SendGrid."""
    subject, html_body = build_html_email(ticker, name, price, target, analysis, articles)

    try:
        response = requests.post(
            "https://api.sendgrid.com/v3/mail/send",
            headers={
                "Authorization": f"Bearer {SENDGRID_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "personalizations": [{"to": [{"email": EMAIL_RECEIVER}]}],
                "from": {"email": EMAIL_SENDER},
                "subject": subject,
                "content": [{"type": "text/html", "value": html_body}],
            },
            timeout=10
        )

        if response.status_code == 202:
            print(f"  📧 Alert email sent to {EMAIL_RECEIVER}")
            return True
        else:
            print(f"  [ERROR] SendGrid error {response.status_code}: {response.text}")
            return False

    except Exception as e:
        print(f"  [ERROR] Failed to send email: {e}")
        return False


def log_price(ticker: str, price: float, alert_below: float):
    """Append price to CSV log file."""
    if not LOG_FILE:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a") as f:
        if not file_exists:
            f.write("timestamp,ticker,price,alert_below,triggered\n")
        triggered = "YES" if price < alert_below else ""
        f.write(f"{timestamp},{ticker},{price},{alert_below},{triggered}\n")


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Stock Price Tracker — starting run ({len(WATCHLIST)} stocks)")
    print("─" * 54)

    triggered_count = 0

    for stock in WATCHLIST:
        ticker      = stock["ticker"]
        name        = stock["name"]
        alert_below = stock["alert_below"]
        news_query  = stock["news_query"]

        price = get_price(ticker)

        if price is None:
            print(f"  {ticker}: ⚠️  Could not retrieve price, skipping.")
            continue

        status = "🔴 BELOW TARGET" if price < alert_below else "✅ above target"
        print(f"  {ticker}: ${price:.2f}  (alert: ${alert_below:.2f})  —  {status}")

        log_price(ticker, price, alert_below)

        if price < alert_below:
            triggered_count += 1
            print(f"  ⚠️  Triggered! Fetching news and running AI analysis...")

            articles = get_news(news_query)
            print(f"  📰 Found {len(articles)} news articles")

            print(f"  🤖 Asking Claude for analysis...")
            analysis = analyze_with_claude(ticker, name, price, alert_below, articles)
            print(f"  ✅ Analysis complete")

            send_smart_email(ticker, name, price, alert_below, analysis, articles)

        print()

    print("─" * 54)
    print(f"Run complete. {triggered_count}/{len(WATCHLIST)} alerts triggered. Exiting.")


if __name__ == "__main__":
    main()
