
import React, { useCallback, useState } from 'react';
import { UploadIcon } from './icons';

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  isLoading: boolean;
  loadingMessage: string;
  error: string | null;
}

export const FileUpload: React.FC<FileUploadProps> = ({ onFileSelect, isLoading, loadingMessage, error }) => {
  const [isDragging, setIsDragging] = useState(false);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      onFileSelect(file);
    }
  };
  
  const handleDrop = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);
    const file = event.dataTransfer.files?.[0];
    if (file && file.type === 'application/pdf') {
      onFileSelect(file);
    }
  }, [onFileSelect]);

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
  };
  
  const handleDragEnter = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(true);
  };
  
  const handleDragLeave = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);
  };

  if (isLoading) {
    return (
      <div className="flex-grow flex flex-col items-center justify-center text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-brand-primary mb-4"></div>
        <p className="text-xl font-semibold text-brand-text">{loadingMessage}</p>
        <p className="text-brand-text-secondary">Please wait, this may take a moment...</p>
      </div>
    );
  }

  return (
    <div className="flex-grow flex flex-col items-center justify-center text-center p-4">
      <div 
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        className={`w-full max-w-2xl p-10 border-2 border-dashed rounded-xl transition-colors duration-300 ${isDragging ? 'border-brand-primary bg-brand-surface' : 'border-brand-border hover:border-brand-primary'}`}
      >
        <input
          type="file"
          id="file-upload"
          className="hidden"
          accept=".pdf"
          onChange={handleFileChange}
        />
        <label htmlFor="file-upload" className="cursor-pointer flex flex-col items-center gap-4">
          <UploadIcon className="w-16 h-16 text-brand-text-secondary"/>
          <h2 className="text-2xl font-bold text-brand-text">Drop your PDF here</h2>
          <p className="text-brand-text-secondary">or click to browse your files</p>
        </label>
      </div>
      {error && (
        <div className="mt-6 bg-red-900/50 border border-red-700 text-red-300 px-4 py-3 rounded-lg max-w-2xl w-full">
          <p className="font-bold">Error</p>
          <p>{error}</p>
        </div>
      )}
      <div className="mt-8 text-center max-w-2xl text-brand-text-secondary text-sm">
        <p>Your document is processed locally in your browser. No data is sent to our servers until you start chatting.</p>
      </div>
    </div>
  );
};
