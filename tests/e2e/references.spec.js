const { test, expect } = require('@playwright/test');

test('Volatility Forecasts Part 3 renders references', async ({ page }) => {
    await page.goto('/posts/volatility-forecasts-3/index.html');

    const refs = page.locator('#refs');
    await expect(refs).toBeVisible();

    // Ensure the bibliography contains the two expected entries.
    await expect(page.locator('#ref-taylor2004')).toBeVisible();
    await expect(page.locator('#ref-liu2020')).toBeVisible();
});
