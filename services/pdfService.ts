
import * as pdfjsLib from 'pdfjs-dist';

// Set worker source for pdf.js
// This is crucial for it to work in a web environment.
// We provide the full CDN URL to the worker script to avoid path resolution issues.
pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://aistudiocdn.com/pdfjs-dist@5.4.296/build/pdf.worker.min.mjs';

export async function parsePdf(file: File): Promise<string> {
  const fileReader = new FileReader();

  return new Promise((resolve, reject) => {
    fileReader.onload = async (event) => {
      if (!event.target?.result) {
        return reject(new Error("Failed to read file."));
      }

      try {
        const typedarray = new Uint8Array(event.target.result as ArrayBuffer);
        const pdf = await pdfjsLib.getDocument({ data: typedarray }).promise;
        let fullText = '';

        for (let i = 1; i <= pdf.numPages; i++) {
          const page = await pdf.getPage(i);
          const textContent = await page.getTextContent();
          const pageText = textContent.items.map(item => 'str' in item ? item.str : '').join(' ');
          fullText += pageText + '\n\n';
        }
        
        resolve(fullText);

      } catch (error) {
        console.error('Error parsing PDF:', error);
        reject(new Error('Could not parse the PDF file. It might be corrupted or protected.'));
      }
    };

    fileReader.onerror = (error) => {
      reject(new Error('Error reading file: ' + error));
    };

    fileReader.readAsArrayBuffer(file);
  });
}