/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#0b0e14",
        surface: "#10131a",
        card: "#161a21",
        cardHover: "#22262f",
        border: "rgba(69, 72, 79, 0.15)",
        primary: "#97a9ff",
        secondary: "#bf81ff",
        tertiary: "#8ff5ff",
        textMain: "#ecedf6",
        textMuted: "#a9abb3"
      },
      fontFamily: {
        display: ['"Space Grotesk"', 'sans-serif'],
        body: ['Inter', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
      backgroundImage: {
        'gradient-primary': 'linear-gradient(135deg, #97a9ff, #bf81ff)',
      }
    },
  },
  plugins: [],
}
