import React, { useState, useCallback, useEffect } from 'react';
import { FileUpload } from './components/FileUpload';
import { ChatView } from './components/ChatView';
import { Header } from './components/Header';
import { parsePdf } from './services/pdfService';
import { uploadPdf, createChatSession } from './services/geminiService';
import type { ChatSession } from './services/geminiService';

export default function App() {
  const [chatSession, setChatSession] = useState<ChatSession | null>(null);
  const [fileName, setFileName] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [loadingMessage, setLoadingMessage] = useState<string>('');
  const [error, setError] = useState<string | null>(null);

  const handleFileSelect = useCallback(async (file: File) => {
    if (!file) {
      setError('No file selected.');
      return;
    }

    setIsLoading(true);
    setLoadingMessage('Uploading and processing PDF...');
    setError(null);
    setChatSession(null);

    try {
      setFileName(file.name);
      
      setLoadingMessage('Parsing PDF...');
      const text = await parsePdf(file);

      setLoadingMessage('Uploading and processing text...');
      const doc_id = await uploadPdf(text);
      sessionStorage.setItem('doc_id', doc_id);
      sessionStorage.setItem('fileName', file.name);
      
      setLoadingMessage('Initializing AI chat...');
      const session = await createChatSession(doc_id);
      setChatSession(session);

    } catch (e) {
      const errorMessage = e instanceof Error ? e.message : 'An unknown error occurred.';
      console.error(e);
      setError(`Failed to process PDF: ${errorMessage}`);
    } finally {
      setIsLoading(false);
      setLoadingMessage('');
    }
  }, []);

  const handleReset = () => {
    sessionStorage.removeItem('doc_id');
    sessionStorage.removeItem('fileName');
    setChatSession(null);
    setFileName('');
    setError(null);
  };

  useEffect(() => {
    const cachedDocId = sessionStorage.getItem('doc_id');
    const cachedFileName = sessionStorage.getItem('fileName');
    if (cachedDocId && cachedFileName) {
      setFileName(cachedFileName);
      createChatSession(cachedDocId).then(setChatSession);
    }
  }, []);

  return (
    <div className="flex flex-col h-screen bg-brand-bg font-sans">
      <Header fileName={fileName} onNewChat={handleReset} />
      <main className="flex-grow container mx-auto p-4 flex flex-col overflow-hidden">
        {chatSession ? (
          <ChatView chatSession={chatSession} />
        ) : (
          <FileUpload 
            onFileSelect={handleFileSelect} 
            isLoading={isLoading} 
            loadingMessage={loadingMessage} 
            error={error} 
          />
        )}
      </main>
    </div>
  );
}