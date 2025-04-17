import React from 'react';
import { Button } from "@heroui/button";
import { Progress } from "@heroui/progress"; // Or a Spinner component if available

interface RecorderControlsProps {
    isRecording: boolean;
    isProcessing: boolean;
    onToggleRecording: () => void;
}

const RecorderControls: React.FC<RecorderControlsProps> = ({
    isRecording,
    isProcessing,
    onToggleRecording
}) => {
    return (
        <div className="flex justify-center items-center gap-4 mb-4">
            <Button
                onPress={onToggleRecording}
                isDisabled={isProcessing}
                color={isRecording ? "danger" : "primary"}
                size="lg"
                startContent={
                    isRecording ? (
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                             <rect x="6" y="6" width="12" height="12" />
                        </svg>
                    ) : (
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                            <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5-3c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>
                        </svg>
                    )
                }
            >
                {isRecording ? "Stop Recording" : "Start Recording"}
            </Button>
            {isProcessing && (
                 <Progress
                     size="sm"
                     isIndeterminate
                     aria-label="Processing..."
                     className="max-w-xs"
                 />
                // Alternative: Use a Spinner component if available
                // <Spinner size="md" label="Processing..." />
            )}
        </div>
    );
};

export default RecorderControls; 