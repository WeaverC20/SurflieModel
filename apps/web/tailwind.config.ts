import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        'ocean-dark': '#0a1929',
        'ocean-blue': '#0d47a1',
        'ocean-light': '#1976d2',
      },
    },
  },
  plugins: [],
}
export default config
