# Sources and attribution

The generated-program policy and fix-loop workflow are adapted from NVIDIA ASPIRE,
revision fa00f0ef2bc1d91d594b9aa502c22133c9f538f7:
https://github.com/NVlabs/ASPIRE

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
ASPIRE's Apache-2.0 license and upstream notice are preserved unchanged in
third_party/ASPIRE-LICENSE and third_party/ASPIRE-NOTICE. Changes here adapt the
public-API program contract, roles and evaluation protocol to RoboDojo and a
persistent dual-loop runtime; this is not an unchanged ASPIRE benchmark.

RoboDojo compatibility behavior is derived from the owner's EmbodiedDualLoop
revision 01210eedc8b9424e15b0d1b0e1e115b026dc89c8. Upstream simulator sources,
assets and licenses remain external, are not redistributed, and must be installed
and pinned by the deployment operator. No third-party robot assets are included.

SAM3 is an external dependency from https://github.com/facebookresearch/sam3.
No SAM3 source, credentials or checkpoint is bundled. Operators must obtain
authorized weights and comply with SAM3's own code/checkpoint terms. The service
records the actual implementation and checkpoint hashes; local fixture tests are
not evidence of actual SAM3 recognition or legal authorization to obtain weights.
