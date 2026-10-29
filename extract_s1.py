"""Download Anthropic, PBC's Form S-1 (filed 2026-10-26) from SEC EDGAR and extract the Risk Factors section.
Writes s1.txt (whole prospectus, plain text) and risk_factors.txt (printed pages 27-120)."""
import re, html, urllib.request
URL = "https://www.sec.gov/Archives/edgar/data/2130357/000162828026067168/anthropicpbc-sx1.htm"
req = urllib.request.Request(URL, headers={"User-Agent": "flybrain-ipo research (set your own contact email here)"})
s = urllib.request.urlopen(req).read().decode("utf-8", "ignore")
t = re.sub(r"<(script|style).*?</\1>", "", s, flags=re.S)
t = re.sub(r"<br\s*/?>|</p>|</div>|</tr>", "\n", t)
t = html.unescape(re.sub(r"<[^>]+>", "", t))
t = re.sub(r"[ \t\xa0]+", " ", t); t = re.sub(r"\n\s*\n+", "\n", t)
open("s1.txt", "w").write(t)
a = t.index("Investing in our Class A common stock involves a high degree of risk")
b = t.index("Special Note Regarding Forward-looking \nStatements", a) if "Special Note Regarding Forward-looking \nStatements" in t[a:] else t.index("Special Note Regarding Forward-looking", a)
rf = re.sub(r"\n\d{1,3}\nTable of Contents\n", "\n", t[a:b])
open("risk_factors.txt", "w").write(rf)
print(len(re.findall(r"[A-Za-z][A-Za-z'’\-]*", rf)), "words of risk factors")
