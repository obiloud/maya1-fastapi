import multiprocessing as mp
import numpy as np
import os
import logging
from .constants import SNAC_MODEL_NAME
from .logging import setup_child_logging
from multiprocessing.synchronize import Event

class AsyncSNACProcess(mp.Process):
    def __init__(self, input_queue: mp.Queue, output_queue: mp.Queue, log_queue: mp.Queue, device: str = "cpu", ready_event: Event = None):
        # Explicitly call the super constructor
        super(AsyncSNACProcess, self).__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.device = device
        self.log_queue = log_queue
        self.ready_event = ready_event
        # Daemon ensures the child process exits when the main process does
        self.daemon = True 

    def run(self):
        """
        The entry point for the process. 
        Imports and model loading MUST happen here.
        """

        setup_child_logging(self.log_queue)
        logger = logging.getLogger("async_snac_process")
        logger.info(f"Child process {os.getpid()} initialized.")
        
        try:
            # Local imports to avoid pickling issues
            from snac import SNAC
            import torch
            
            # Constants must be available in this scope
            CODE_TOKEN_OFFSET = 128266
            CODE_END_TOKEN_ID = 156937

            logger.info(f"[SNAC Process {os.getpid()}] Initializing SNAC on {self.device}...")
            # Load model inside the child process
            snac_model = os.environ.get('SNAC_MODEL_PATH', SNAC_MODEL_NAME)
            model = SNAC.from_pretrained(snac_model).eval().to(self.device)
            
            if self.device == "cpu":
                torch.set_num_threads(2)
                torch.set_grad_enabled(False)

            logger.info(f"[SNAC Process] Ready and listening for tokens.")

            with torch.inference_mode():
                dummy_codes = [
                    torch.zeros((1, 1), dtype=torch.long, device=self.device),
                    torch.zeros((1, 2), dtype=torch.long, device=self.device),
                    torch.zeros((1, 4), dtype=torch.long, device=self.device)
                ]
                _ = model.decoder(model.quantizer.from_codes(dummy_codes))
            
            self.ready_event.set() 
            logger.info("✅ SNAC Decoder is fully warmed up and signaled READY.")

            while True:
                # Use a timeout to keep the process responsive to exit signals
                try:
                    item = self.input_queue.get(timeout=1.0)
                except: # queue.Empty
                    continue

                if item is None: # Sentinel for shutdown
                    logger.info("Shutdown sentinel received. Exiting SNAC process.")
                    break
                
                tokens = item
                
                try:
                    # Logic for decoding
                    if len(tokens) % 7 != 0:
                        # Truncate to the nearest frame to prevent hierarchy corruption
                        tokens = tokens[:(len(tokens) // 7) * 7]

                    arr = (np.array(tokens).reshape(-1, 7) - 128266) % 4096
                    
                    with torch.inference_mode():
                        # Level 1: [1, seq_len]
                        l1 = torch.from_numpy(arr[:, 0]).to(self.device).long().unsqueeze(0)
                        
                        # Level 2: [1, seq_len * 2]
                        # We use contiguous() to ensure memory is linear before passing to the model
                        l2 = torch.from_numpy(arr[:, [1, 4]]).to(self.device).long().reshape(1, -1).contiguous()
                        
                        # Level 3: [1, seq_len * 4]
                        l3 = torch.from_numpy(arr[:, [2, 3, 5, 6]]).to(self.device).long().reshape(1, -1).contiguous()
                        
                        # Reconstruct acoustic features
                        z_q = model.quantizer.from_codes([l1, l2, l3])
                        audio = model.decoder(z_q)[0, 0].cpu().numpy()
                    
                    # Non-blocking put to prevent deadlock if main process hangs
                    try:
                        self.output_queue.put_nowait(audio)
                    except mp.queues.Full:
                        logger.warning("Audio output queue full, dropping chunk.")
                        
                except Exception as chunk_error:
                        logger.error(f"Error decoding chunk: {chunk_error}", exc_info=True)

        except Exception as e:
            logger.error(f"[SNAC Process] Runtime Error: {e}")