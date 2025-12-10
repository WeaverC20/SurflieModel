import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import '../styles/globals.css'
import 'mapbox-gl/dist/mapbox-gl.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Surflie - Ocean Current Visualization',
  description: 'Interactive ocean current forecasts for California coast',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>{children}</body>
    </html>
  )
}
