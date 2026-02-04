const { test, expect } = require('@playwright/test');

async function visibleTitles(page) {
    return await page.$$eval('#listing-listing .quarto-post', (els) => {
        return els
            .filter((el) => el.offsetParent !== null)
            .map((el) => {
                const a = el.querySelector('.listing-title a');
                return (a ? a.textContent : '').trim();
            })
            .filter(Boolean);
    });
}

test('Search filters listings by query', async ({ page }) => {
    await page.goto('/pages/search.html');

    // Wait for Quarto listing to initialize.
    await page.waitForFunction(() => {
        const ql = window['quarto-listings'];
        return ql && ql['listing-listing'];
    });

    const initialTitles = await visibleTitles(page);
    expect(initialTitles.length).toBeGreaterThan(5);

    await page.fill('#post-search', 'XGBSTES');
    await page.waitForTimeout(200);

    const filteredTitles = await visibleTitles(page);
    expect(filteredTitles.length).toBeGreaterThan(0);
    expect(filteredTitles.join('\n')).toContain('Volatility Forecasts (Part 3 - XGBSTES Algorithm 2)');

    await page.fill('#post-search', 'zzzzzzzz-nope');
    await page.waitForTimeout(200);
    const noneTitles = await visibleTitles(page);
    expect(noneTitles.length).toBe(0);

    await page.fill('#post-search', '');
    await page.waitForTimeout(200);
    const restoredTitles = await visibleTitles(page);
    expect(restoredTitles.length).toBeGreaterThan(5);
});
