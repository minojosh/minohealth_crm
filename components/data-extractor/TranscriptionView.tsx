import React from 'react';
import { Textarea } from "@heroui/input";
import { Button } from "@heroui/button";

interface TranscriptionViewProps {
    transcription: string;
    onTranscriptionChange: (value: string) => void;
    onClear: () => void;
    isDisabled: boolean;
}

const TranscriptionView: React.FC<TranscriptionViewProps> = ({
    transcription,
    onTranscriptionChange,
    onClear,
    isDisabled
}) => {
    return (
        <div className="space-y-2">
            <div className="flex justify-between items-center">
                 <h2 className="text-xl font-semibold text-white">Transcription</h2>
                 {transcription && (
                     <Button
                        size="sm"
                        color="danger"
                        variant="flat"
                        onPress={onClear}
                        isDisabled={isDisabled}
                     >
                         Clear
                     </Button>
                 )}
            </div>
            <Textarea
                value={transcription}
                onValueChange={onTranscriptionChange}
                placeholder="Transcription will appear here... or type manually."
                isDisabled={isDisabled}
                rows={8}
                labelPlacement="outside"
                classNames={{
                     inputWrapper: "bg-gray-700 border-gray-600",
                     input: "text-gray-100"
                }}
            />
            <p className="text-gray-400 text-xs">
                {transcription ? `${transcription.length} characters` : 'No transcription yet'}
            </p>
        </div>
    );
};

export default TranscriptionView; 