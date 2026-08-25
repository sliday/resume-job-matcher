import { readFile } from 'node:fs/promises';
import { extractText, getDocumentProxy } from 'unpdf';

export async function extractPdfText(filePath: string): Promise<string> {
  const buffer = await readFile(filePath);
  const pdf = await getDocumentProxy(new Uint8Array(buffer));
  const { text } = await extractText(pdf, { mergePages: true });
  return text.trim();
}
