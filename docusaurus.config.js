// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const lightCodeTheme = require("prism-react-renderer/themes/github");
const darkCodeTheme = require("prism-react-renderer/themes/dracula");
const math = require('remark-math');
const katex = require('rehype-katex');

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: "Was it worth it?",
  tagline: "Empoered with knowledge",
  url: "https://onism.space",
  baseUrl: "/",
  onBrokenLinks: "throw",
  onBrokenMarkdownLinks: "warn",
  favicon: "img/logo.svg",
  organizationName: "Asthestarsfalll", // Usually your GitHub org/user name.
  projectName: "Asthestarsfalll.github.io", // Usually your repo name.


  presets: [
    [
      "@docusaurus/preset-classic",
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve("./sidebars.js"),
          editUrl: "https://github.dev/Asthestarsfalll/Asthestarsfalll.github.io/blob/master/",
          remarkPlugins: [math],
          rehypePlugins: [katex],
        },
        blog: {
          showReadingTime: true,
          editUrl: "https://github.dev/Asthestarsfalll/Asthestarsfalll.github.io/blob/master/",
          blogSidebarTitle: "All posts",
          blogSidebarCount: "ALL",
          remarkPlugins: [math],
          rehypePlugins: [katex],
        },
        theme: {
          customCss: require.resolve("./src/css/custom.css"),
        },
      }),
    ],
  ],

  stylesheets: [
    {
      href: 'https://fastly.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
      type: 'text/css',
      integrity:
        'sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM',
      crossorigin: 'anonymous',
    },
  ],

  plugins: [
    [
      require.resolve("@easyops-cn/docusaurus-search-local"),
      {
        // ... Your options.
        // `hashed` is recommended as long-term-cache of index file is possible.
        hashed: true,
        // For Docs using Chinese, The `language` is recommended to set to:
        // ```
        language: ["en", "zh"],
        // ```
        // When applying `zh` in language, please install `nodejieba` in your project.
        translations: {
          search_placeholder: "Search",
          see_all_results: "See all results",
          no_results: "No results.",
          search_results_for: 'Search results for "{{ keyword }}"',
          search_the_documentation: "Search the documentation",
          count_documents_found: "{{ count }} document found",
          count_documents_found_plural: "{{ count }} documents found",
          no_documents_were_found: "No documents were found",
        },
      },
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      navbar: {
        title: "Onism Space",
        logo: {
          alt: "Logo",
          src: "img/logo.svg",
        },
        items: [
          {
            to: "/blog",
            label: "Blogs",
            position: "left"
          },
          {
            to: "/docs/notes",
            label: "Notes",
            position: "left"
          },
          {
            // type: "doc",
            // docId: "index",
            to: "/docs/meaningless",
            position: "left",
            label: "Meaningless",
          },
          {
            // href: "https://ExCore.github.io",
            to: "/docs/excore",
            label: "ExCore",
            position: "left",
          },
          // {
          //   // href: "https://MegBox.github.io",
          //   to: "/docs/megbox",
          //   label: "MegBox",
          //   position: "left",
          // },
          {
            href: "/about",
            label: "About & Links",
            position: "right",
          },
          {
            href: "https://github.com/Asthestarsfalll",
            label: "Me on GitHub",
            position: "right",
          },
        ],
      },
      footer: {
        style: "light",
        links: [{
          title: "Others",
          items: [{
            label: "Meaningless Corner",
            to: "/docs/meaningless/index",
          },],
        },
        {
          title: "Community",
          items: [{
            label: "NEET CV",
            href: "https://github.com/neet-cv",
          },
          {
            label: "sanyuankexie",
            href: "https://github.com/sanyuankexie",
          },
          ],
        },
        {
          title: "More",
          items: [{
            label: "excore.github.io",
            to: "https://excore.github.io",
          },
          {
            label: "See me on Github",
            href: "https://github.com/Asthestarsfalll",
          },
          {
            label: "Powered by docusaurus",
            href: "https://github.com/facebook/docusaurus",
          },
          ],
        },
        ],
        copyright: `<a href="https://github.com/Asthestarsfalll" target="_blank">@Asthestarsfalll</a> ${new Date().getFullYear()} all rights reserved `,
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
        defaultMode: 'dark'
      },
    }),
};

module.exports = config;
