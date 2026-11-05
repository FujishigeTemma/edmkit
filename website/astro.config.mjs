// @ts-check
import starlight from "@astrojs/starlight";
import { defineConfig } from "astro/config";

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
          label: "Examples",
          items: [{ autogenerate: { directory: "examples" } }],
        },
        { label: "Pitfalls", slug: "pitfalls" },
        { label: "Benchmarks", slug: "benchmarks" },
        {
          label: "API Reference",
          items: [{ autogenerate: { directory: "reference" } }],
        },
      ],
      editLink: {
        baseUrl: "https://github.com/FujishigeTemma/edmkit/edit/main/website/",
      },
      lastUpdated: true,
    }),
  ],
});
