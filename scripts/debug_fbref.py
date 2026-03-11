#!/usr/bin/env python3
"""Debug: analyze FBref passing page data availability."""
import undetected_chromedriver as uc
import pandas as pd
import io
import time

options = uc.ChromeOptions()
driver = uc.Chrome(options=options, version_main=145)

try:
    url = "https://fbref.com/en/comps/9/2023-2024/passing/2023-2024-Premier-League-Stats"
    driver.get(url)

    for _ in range(30):
        if "Just a moment" not in driver.title:
            break
        time.sleep(2)
    print(f"Page title: {driver.title}")
    time.sleep(5)

    from selenium.webdriver.common.by import By

    # Check what tables exist
    tables = driver.find_elements(By.CSS_SELECTOR, "table.stats_table")
    print(f"stats_table elements: {len(tables)}")

    # Get the player-level table (largest one)
    for i, t in enumerate(tables):
        html = t.get_attribute("outerHTML")
        dfs = pd.read_html(io.StringIO(html))
        if dfs:
            df = dfs[0]
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ["_".join(str(x) for x in col).strip("_") for col in df.columns]
            print(f"\nTable {i}: {len(df)} rows x {len(df.columns)} cols")
            if len(df) > 100:
                # This is the player table
                print(f"  Columns: {list(df.columns)}")
                print(f"\n  First 3 rows sample:")
                for col in df.columns[:15]:
                    vals = df[col].head(3).tolist()
                    nn = df[col].notna().mean()
                    print(f"    {col}: {vals} (non-null: {nn:.1%})")

    # Try using JavaScript to get data from the page
    print("\n--- JS data extraction ---")
    # Check if FBref stores data in any global variable
    result = driver.execute_script("""
        var tables = document.querySelectorAll('table.stats_table');
        var info = [];
        tables.forEach(function(t, i) {
            var rows = t.querySelectorAll('tbody tr');
            var firstRow = rows[0];
            if (firstRow) {
                var cells = firstRow.querySelectorAll('td');
                var cellData = [];
                cells.forEach(function(c) {
                    cellData.push({
                        stat: c.getAttribute('data-stat'),
                        text: c.textContent.trim(),
                        innerHTML: c.innerHTML.trim().substring(0, 50)
                    });
                });
                info.push({table: i, rowCount: rows.length, cells: cellData});
            }
        });
        return JSON.stringify(info);
    """)
    import json
    data = json.loads(result)
    for table_info in data:
        print(f"\nTable {table_info['table']}: {table_info['rowCount']} rows")
        for cell in table_info['cells'][:15]:
            print(f"  data-stat={cell['stat']}: text={cell['text']!r}, html={cell['innerHTML']!r}")

    # Also check: does the Share/Export CSV feature exist?
    csv_links = driver.find_elements(By.CSS_SELECTOR, "a[href*='share']")
    print(f"\nShare links: {len(csv_links)}")

    # Check for the CSV export button
    buttons = driver.find_elements(By.CSS_SELECTOR, "button.tooltip")
    for b in buttons[:5]:
        print(f"Button: {b.text!r}, tip={b.get_attribute('tip')}")

finally:
    try:
        driver.quit()
    except:
        pass
