const { test, expect } = require('@playwright/test');
const { execFileSync } = require('node:child_process');
const { existsSync, readdirSync } = require('node:fs');
const path = require('node:path');

test('render scope includes only public site sources', async () => {
    const project = JSON.parse(execFileSync('quarto', ['inspect'], { encoding: 'utf8' }));
    const inputs = project.files.input.map((file) => (
        path.relative(process.cwd(), file).split(path.sep).join('/')
    ));
    const expected = [
        'index.qmd',
        ...readdirSync('pages').filter((file) => file.endsWith('.qmd')).map((file) => `pages/${file}`),
        ...readdirSync('posts', { withFileTypes: true })
            .filter((entry) => entry.isDirectory())
            .flatMap((entry) => ['index.qmd', 'index.ipynb']
                .map((file) => `posts/${entry.name}/${file}`)
                .filter((file) => existsSync(file))),
    ];

    expect(inputs.sort()).toEqual(expected.sort());
});

test('Volatility Forecasts Part 3 renders references', async ({ page }) => {
    await page.goto('/posts/volatility-forecasts-3/index.html');

    const refs = page.locator('#refs');
    await expect(refs).toBeVisible();

    // Ensure the bibliography contains the two expected entries.
    await expect(page.locator('#ref-taylor2004')).toBeVisible();
    await expect(page.locator('#ref-liu2020')).toBeVisible();
});
