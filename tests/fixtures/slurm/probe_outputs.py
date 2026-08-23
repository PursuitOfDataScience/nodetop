"""Verbatim ``sbatch --test-only`` output, captured from a production cluster.

These are the real strings, not paraphrases.  The parser's job is to be right
about *these*, so they are kept exactly as the controller emitted them --
including the inconsistent ``sbatch: error:`` prefixing and the unprefixed
``allocation failure:`` line.
"""

# Accepted.  Note the QOS the controller selected is not the one implied by
# the partition name: "beagle3" was auto-promoted to "beagle3-prio".
ACCEPTED = """sbatch: Verify job submission ...
sbatch: Partition: beagle3
sbatch: QOS-Flag: beagle3-prio
sbatch: Account: rcc-staff
sbatch: Verification: ***PASSED***
sbatch: Job 54116041 to start at 2026-08-21T17:00:12 using 4 processors on nodes beagle3-0006 in partition beagle3
"""

# The site plugin refuses outright.  This is the case sacctmgr does not
# predict: the association table lists the account, the submit filter does not
# honour it.
PLUGIN_REJECTED = """sbatch: error: Verify job submission ...
sbatch: error: Partition: pedramh-gpu
sbatch: error: QOS-Flag: pedramh-gpu
sbatch: error: Account: pi-pedramh
sbatch: error: Verification: ***REJECTED***
sbatch: error: Reason: Invalid membership to account [pi-pedramh]
allocation failure: Access/permission denied
"""

# The dangerous one: the plugin says PASSED and Slurm core refuses anyway.
# Anything that greps for "Verification:" and stops reads this as success.
PLUGIN_PASSED_CORE_REFUSED = """sbatch: error: Verify job submission ...
sbatch: error: Partition: pedramh-gpu
sbatch: error: QOS-Flag: pedramh-gpu
sbatch: error: Account: rcc-staff
sbatch: error: Verification: ***PASSED***
allocation failure: Invalid account or account/partition combination specified
"""

NO_ACCOUNT = """sbatch: error: Verify job submission ...
sbatch: error: Partition: beagle3
sbatch: error: QOS-Flag: beagle3
sbatch: error: Account: unknown
sbatch: error: Verification: ***REJECTED***
sbatch: error: Reason: Account is not specified
allocation failure: Access/permission denied
"""

SHAPE_TOO_BIG = """sbatch: error: Verify job submission ...
sbatch: error: Partition: beagle3
sbatch: error: QOS-Flag: beagle3-prio
sbatch: error: Account: rcc-staff
sbatch: error: Verification: ***PASSED***
allocation failure: Node count specification invalid
"""

SHARED_PARTITION = """sbatch: Using a shared partition ...
sbatch: Partition: gpu
sbatch: QOS-Flag: gpu
sbatch: Account: rcc-staff
sbatch: Verification: ***PASSED***
sbatch: Job 54116043 to start at 2026-08-22T19:52:18 using 4 processors on nodes midway3-0278 in partition gpu
"""

# A controller that cannot write job scripts fails EVERY submission, including
# a bare --wrap=hostname.  Misreading this as an access problem sends you
# hunting for a permission that was never missing.
CONTROLLER_IO_ERROR = """sbatch: error: Batch job submission failed: I/O error writing script/environment to file
"""

# Stock Slurm phrasings, for clusters with no site plugin.
STOCK_QOS_VIOLATION = """sbatch: error: Batch job submission failed: Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)
"""

STOCK_BAD_PARTITION = """sbatch: error: Batch job submission failed: Invalid partition name specified
"""

STOCK_NODE_CONFIG = """sbatch: error: Batch job submission failed: Requested node configuration is not available
"""

STOCK_TIME_LIMIT = """sbatch: error: Batch job submission failed: Requested time limit is invalid (missing or exceeds some limit)
"""
