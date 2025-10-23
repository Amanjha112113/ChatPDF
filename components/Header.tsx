
import React from 'react';
import { FilePlusIcon } from './icons';

interface HeaderProps {
    fileName: string;
    onNewChat: () => void;
}

export const Header: React.FC<HeaderProps> = ({ fileName, onNewChat }) => {
    return (
        <header className="bg-brand-surface border-b border-brand-border shadow-md">
            <div className="container mx-auto p-4 flex justify-between items-center">
                <div className="flex items-center gap-3">
                     <h1 className="text-xl font-bold text-brand-text">Chat with PDF</h1>
                     {fileName && (
                         <span className="hidden md:inline-block bg-brand-bg border border-brand-border text-brand-text-secondary text-sm rounded-md px-3 py-1 truncate max-w-xs">
                             {fileName}
                         </span>
                     )}
                </div>
                <button
                    onClick={onNewChat}
                    className="flex items-center gap-2 bg-brand-primary text-white font-semibold py-2 px-4 rounded-lg hover:bg-brand-primary-hover transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-brand-primary focus:ring-opacity-50"
                >
                    <FilePlusIcon className="w-5 h-5" />
                    <span className="hidden sm:inline">New Chat</span>
                </button>
            </div>
        </header>
    );
};
