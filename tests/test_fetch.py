from core import fetch

SAMPLE_HTML = b"""
<html>
  <body>
    <nav>This is navigation</nav>
    <article>
      <h1>Accurate Simulation</h1>
      <p>We present a novel barrier method for collision handling.</p>
      <p>It improves accuracy and stability for cloth.</p>
    </article>
    <div class="cookie">We need your consent to use cookies.</div>
  </body>
</html>
"""


def test_clean_html_filters_noise():
    text = fetch._clean_html(SAMPLE_HTML)
    assert "Accurate Simulation" in text
    assert "cookie" not in text.lower()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    assert len(lines) == len(set(line.lower() for line in lines))
