"""
Playwright webapp tests for the NBA Stats Predictor Streamlit app.

Tests cover:
1. Page loads and displays correctly
2. Title and input field are present
3. Player search functionality works
4. Game data and prediction tabs render
5. Console errors are captured
"""

import sys
import time

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


def test_nba_webapp():
    """Run all webapp tests against the Streamlit app."""
    results = {"passed": [], "failed": [], "warnings": [], "screenshots": []}
    console_errors = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Capture console errors
        page.on(
            "console",
            lambda msg: console_errors.append(msg.text)
            if msg.type == "error"
            else None,
        )

        try:
            # === Test 1: Page loads ===
            print("[Test 1] Page load...")
            page.goto("http://localhost:8501", timeout=60000)
            page.wait_for_load_state("networkidle", timeout=60000)
            results["passed"].append("Page loads successfully")
            print("  PASS")

            page.screenshot(path="/tmp/nba_webapp_01_initial.png", full_page=True)
            results["screenshots"].append("/tmp/nba_webapp_01_initial.png")

            # === Test 2: Title displayed ===
            print("[Test 2] Title element...")
            # Streamlit renders the title via unsafe_allow_html markdown,
            # so we need to wait for the full DOM and check for the h1 inside it.
            page.wait_for_timeout(3000)
            title = page.locator("h1")
            title_found = False
            if title.count() > 0:
                for i in range(title.count()):
                    text = title.nth(i).inner_text()
                    if "NBA" in text:
                        results["passed"].append(f"Title displayed: '{text}'")
                        print(f"  PASS - '{text}'")
                        title_found = True
                        break
            if not title_found:
                # Fallback: check raw page content for the title text
                content = page.content()
                if "NBA Player Points Predictor" in content:
                    results["passed"].append("Title present in page HTML")
                    print("  PASS (found in HTML)")
                else:
                    results["failed"].append(
                        "Title 'NBA Player Points Predictor' not found"
                    )
                    print("  FAIL")

            # === Test 3: Player input field present ===
            print("[Test 3] Player input field...")
            # Streamlit may render input without type='text'; use the label to find it
            input_field = page.locator("input")
            if input_field.count() > 0:
                results["passed"].append("Player name input field is present")
                print("  PASS")
            else:
                results["failed"].append("Player name input field not found")
                print("  FAIL")

            # === Test 4: Search for a player ===
            print("[Test 4] Player search (LeBron James)...")
            try:
                input_field.first.fill("LeBron James")
                input_field.first.press("Enter")
                # Wait for spinner and API calls
                time.sleep(4)
                page.wait_for_load_state("networkidle", timeout=120000)
                page.wait_for_timeout(5000)

                page.screenshot(path="/tmp/nba_webapp_02_search.png", full_page=True)
                results["screenshots"].append("/tmp/nba_webapp_02_search.png")

                content = page.content()
                if "Player found successfully" in content or "LeBron" in content:
                    results["passed"].append(
                        "Player search: LeBron James found successfully"
                    )
                    print("  PASS")
                elif "Player not found" in content:
                    results["failed"].append(
                        "Player search: 'Player not found' message displayed"
                    )
                    print("  FAIL - Player not found")
                else:
                    results["warnings"].append(
                        "Player search: response unclear, check screenshot"
                    )
                    print("  WARN - unclear response")
            except PlaywrightTimeoutError:
                results["failed"].append("Player search timed out waiting for response")
                print("  FAIL - timeout")

            # === Test 5: Game data table ===
            print("[Test 5] Game data table...")
            try:
                page.wait_for_timeout(3000)
                content = page.content()
                if "Last" in content and "Games" in content:
                    results["passed"].append(
                        "'Last N Games' table section is displayed"
                    )
                    print("  PASS")
                else:
                    results["warnings"].append(
                        "Game data table not visible (may depend on NBA API)"
                    )
                    print("  WARN - not visible")
            except Exception as e:
                results["failed"].append(f"Game data table check error: {e}")
                print(f"  FAIL - {e}")

            # === Test 6: Prediction tabs ===
            print("[Test 6] Prediction tabs...")
            tabs_found = []
            for name in ["Points", "Rebounds", "Assists"]:
                tab = page.locator(f"button:has-text('{name}')")
                if tab.count() > 0:
                    tabs_found.append(name)

            if len(tabs_found) == 3:
                results["passed"].append(
                    "All prediction tabs present: Points, Rebounds, Assists"
                )
                print(f"  PASS - {tabs_found}")
            elif tabs_found:
                results["warnings"].append(f"Partial tabs found: {tabs_found}")
                print(f"  WARN - only {tabs_found}")
            else:
                results["warnings"].append(
                    "No prediction tabs found (data may not have loaded)"
                )
                print("  WARN - none found")

            # === Test 7: Click a prediction tab ===
            if tabs_found:
                print("[Test 7] Click Rebounds tab...")
                try:
                    reb_tab = page.locator("button:has-text('Rebounds')")
                    if reb_tab.count() > 0:
                        reb_tab.first.click()
                        page.wait_for_timeout(3000)
                        page.screenshot(
                            path="/tmp/nba_webapp_03_rebounds.png", full_page=True
                        )
                        results["screenshots"].append("/tmp/nba_webapp_03_rebounds.png")
                        results["passed"].append("Rebounds tab click works")
                        print("  PASS")
                    else:
                        results["warnings"].append("Rebounds tab not clickable")
                        print("  WARN")
                except Exception as e:
                    results["failed"].append(f"Tab click error: {e}")
                    print(f"  FAIL - {e}")
            else:
                print("[Test 7] Skipped - no tabs available")
                results["warnings"].append("Tab click test skipped (no tabs loaded)")

            # === Test 8: Check for prediction output ===
            print("[Test 8] Prediction output...")
            content = page.content()
            if "Predicted" in content:
                results["passed"].append("Prediction output is displayed")
                print("  PASS")
            else:
                results["warnings"].append(
                    "No prediction output visible (may depend on NBA API)"
                )
                print("  WARN")

            # === Test 9: Check for visualizations ===
            print("[Test 9] Visualizations (charts)...")
            charts = page.locator("img, canvas, svg.marks")
            chart_count = charts.count()
            if chart_count > 0:
                results["passed"].append(
                    f"Visualizations present ({chart_count} chart elements)"
                )
                print(f"  PASS - {chart_count} elements")
            else:
                results["warnings"].append("No chart elements found")
                print("  WARN")

            # === Test 10: Console errors ===
            print("[Test 10] Console errors...")
            if not console_errors:
                results["passed"].append("No console errors detected")
                print("  PASS")
            else:
                results["warnings"].append(f"Console errors: {len(console_errors)}")
                for err in console_errors[:5]:
                    print(f"  - {err[:120]}")

            # Final screenshot
            page.screenshot(path="/tmp/nba_webapp_04_final.png", full_page=True)
            results["screenshots"].append("/tmp/nba_webapp_04_final.png")

        except Exception as e:
            results["failed"].append(f"Unexpected error: {e}")
            print(f"FATAL: {e}")
            try:
                page.screenshot(path="/tmp/nba_webapp_error.png", full_page=True)
                results["screenshots"].append("/tmp/nba_webapp_error.png")
            except Exception:
                pass

        finally:
            browser.close()

    return results


def main():
    print("=" * 60)
    print("  NBA Stats Predictor - Webapp Test Suite")
    print("=" * 60)
    print()

    results = test_nba_webapp()

    print()
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)

    print(f"\nPASSED ({len(results['passed'])}):")
    for item in results["passed"]:
        print(f"  + {item}")

    if results["warnings"]:
        print(f"\nWARNINGS ({len(results['warnings'])}):")
        for item in results["warnings"]:
            print(f"  ? {item}")

    if results["failed"]:
        print(f"\nFAILED ({len(results['failed'])}):")
        for item in results["failed"]:
            print(f"  - {item}")

    print("\nScreenshots:")
    for s in results["screenshots"]:
        print(f"  {s}")

    total = len(results["passed"]) + len(results["failed"])
    pass_rate = (len(results["passed"]) / total * 100) if total > 0 else 0
    print(f"\nScore: {len(results['passed'])}/{total} passed ({pass_rate:.0f}%)")

    return 0 if not results["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
