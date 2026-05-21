// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

// https://astro.build/config
export default defineConfig({
  site: "https://fujishigetemma.github.io",
  base: "/edmkit",
  integrations: [
    starlight({
      title: "edmkit",
      description: "Simple, fast Empirical Dynamic Modeling library for Python",
      social: [
        { icon: "github", label: "GitHub", href: "https://github.com/FujishigeTemma/edmkit" },
      ],
      sidebar: [
        { label: "Getting Started", slug: "getting-started" },
        {
          label: "Concepts",
          items: [
            { label: "What is EDM?", slug: "concepts/edm" },
            { label: "Time-delay embedding", slug: "concepts/embedding" },
            { label: "Simplex projection", slug: "concepts/simplex-projection" },
            { label: "S-Map", slug: "concepts/smap" },
            { label: "Convergent Cross Mapping", slug: "concepts/ccm" },
          ],
        },
        {
          label: "Guides",
          items: [
            { label: "Choosing E and tau", slug: "guides/choosing-parameters" },
            { label: "Forecasting a time series", slug: "guides/forecasting" },
            { label: "Testing causality with CCM", slug: "guides/causality-with-ccm" },
          ],
        },
        {
          label: "API Reference",
          autogenerate: { directory: "reference" },
        },
      ],
      editLink: {
        baseUrl: "https://github.com/FujishigeTemma/edmkit/edit/main/website/",
      },
      lastUpdated: true,
    }),
  ],
});
