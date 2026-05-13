import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';

// https://astro.build/config
export default defineConfig({
  site: 'https://riceball-15.github.io/ai-learning-blog',
  base: '/ai-learning-blog',
  integrations: [mdx()],
  markdown: {
    shikiConfig: {
      themes: {
        light: 'github-light',
        dark: 'github-dark'
      },
      wrap: true,
      // 启用行号
      transformers: [{
        name: 'add-line-numbers',
        line(node, line) {
          node.properties.class = 'line';
          node.children.unshift({
            type: 'element',
            tagName: 'span',
            properties: { class: 'line-number' },
            children: [{ type: 'text', value: String(line) }]
          });
        }
      }]
    }
  }
});