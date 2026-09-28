import os
# Force HuggingFace to use local cache only — no network requests (cluster has no internet)
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
from src.models.dr_model import RetiNA_Net
import sys

def run_tests():
    print("Running Model Architecture Smoke Tests...")
    dummy_input = torch.randn(2, 3, 512, 512)
    
    ablations = [
        ('baseline', False, False),
        ('msda_only', True, False),
        ('hff_only', False, True),
        ('proposed', True, True)
    ]
    
    for name, use_msda, use_hff in ablations:
        try:
            model = RetiNA_Net(
                use_msda=use_msda, use_hff=use_hff, num_classes=5,
                use_ordinal=True, use_aux_head=True, use_attention_pool=True,
                drop_path_rate=0.1, dropout=0.1
            )
            out = model(dummy_input)
            
            # New model returns dict with 'logits', 'aux_logits', and 'ordinal_logits'
            assert isinstance(out, dict), f"Output should be dict, got {type(out)}"
            assert out['logits'].shape == (2, 5), f"Logits shape mismatch for {name}: expected (2, 5), got {out['logits'].shape}"
            assert out['aux_logits'].shape == (2, 5), f"Aux logits shape mismatch for {name}: expected (2, 5), got {out['aux_logits'].shape}"
            assert out['ordinal_logits'].shape == (2, 4), f"Ordinal logits shape mismatch for {name}: expected (2, 4), got {out['ordinal_logits'].shape}"
            print(f"[OK] {name.upper()}: Logits {out['logits'].shape}, Aux {out['aux_logits'].shape}, Ordinal {out['ordinal_logits'].shape}")
            
            # Test predict method
            probs = model.predict(dummy_input)
            assert probs.shape == (2, 5), f"Probs shape mismatch: {probs.shape}"
            assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5), "Probs should sum to 1"
            print(f"[OK] {name.upper()}: predict() returns valid probabilities")
            
        except Exception as e:
            print(f"[ERROR] {name.upper()} failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Test without ordinal head
    try:
        model = RetiNA_Net(
            use_msda=True, use_hff=True, num_classes=5,
            use_ordinal=False, use_aux_head=True, use_attention_pool=True,
            drop_path_rate=0.1, dropout=0.1
        )
        out = model(dummy_input)
        assert out['ordinal_logits'] is None, "Ordinal logits should be None when use_ordinal=False"
        print(f"[OK] No-ordinal mode: ordinal_logits is None as expected")
    except Exception as e:
        print(f"[ERROR] No-ordinal mode failed: {e}")
        sys.exit(1)
    
    # Test without aux head
    try:
        model = RetiNA_Net(
            use_msda=True, use_hff=True, num_classes=5,
            use_ordinal=True, use_aux_head=False, use_attention_pool=True,
            drop_path_rate=0.1, dropout=0.1
        )
        out = model(dummy_input)
        assert out['aux_logits'] is None, "Aux logits should be None when use_aux_head=False"
        print(f"[OK] No-aux mode: aux_logits is None as expected")
    except Exception as e:
        print(f"[ERROR] No-aux mode failed: {e}")
        sys.exit(1)
    
    print("\nAll architecture tests passed successfully!")

if __name__ == "__main__":
    run_tests()
