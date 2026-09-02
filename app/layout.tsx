import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  metadataBase: new URL('https://eigenfaces-linear-algebra.csprofkgd.chatgpt.site'),
  title: 'Eigenfaces — Face It: It’s Just Linear Algebra',
  description:
    'An interactive visualization of a face reconstructed as an average plus weighted principal components.',
  openGraph: {
    title: 'Eigenfaces',
    description: 'Face It: It’s Just Linear Algebra',
    type: 'website',
    images: [
      {
        url: '/og.png',
        width: 1200,
        height: 630,
        alt: 'Eigenfaces — Face It: It’s Just Linear Algebra',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Eigenfaces',
    description: 'Face It: It’s Just Linear Algebra',
    images: ['/og.png'],
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
