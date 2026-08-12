import { defineConfig } from 'vite'
import path from 'path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  resolve: {
    alias: {
      // Alias @ to the src directory
      '@': path.resolve(__dirname, './src'),
    },
  },

  // File types to support raw imports. Never add .css, .tsx, or .ts files to this.
  assetsInclude: ['**/*.svg', '**/*.csv'],

  build: {
    chunkSizeWarningLimit: 550,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes('node_modules')) return
          if (id.includes('/recharts/')) return 'vendor-recharts'
          if (id.includes('/@mui/')) return 'vendor-mui'
          if (id.includes('/lucide-react/')) return 'vendor-icons'
          if (id.includes('/react-router') || id.includes('/react-router-dom/')) return 'vendor-router'
          if (id.includes('/@radix-ui/')) return 'vendor-radix'
          if (id.includes('/react/') || id.includes('/react-dom/') || id.includes('/scheduler/')) return 'vendor-react'
          if (id.includes('/motion/') || id.includes('/framer-motion')) return 'vendor-motion'
          return 'vendor'
        },
      },
    },
  },
})
