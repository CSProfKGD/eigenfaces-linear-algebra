import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Eigenfaces — Face It: It’s Just Linear Algebra',
  description:
    'An interactive visualization of a face reconstructed as an average plus weighted principal components.',
  openGraph: {
    title: 'Eigenfaces',
    description: 'Face It: It’s Just Linear Algebra',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Eigenfaces',
    description: 'Face It: It’s Just Linear Algebra',
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
