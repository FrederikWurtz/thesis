from master.train import cli
import os
import sys
import subprocess

def main(argv=None):
    cli.main(argv)
    

if __name__ == '__main__':
        # Check if we're running on macOS
    if sys.platform == 'darwin':
        # Check if already running under caffeinate
        if 'CAFFEINATED' not in os.environ:
            print("=" * 60)
            print("Starting caffeinate to prevent system sleep during training")
            print("This ensures full performance even if the screen turns off")
            print("=" * 60)
            
            # Re-run this script under caffeinate
            # -d: Prevent display from sleeping
            # -i: Prevent system from idle sleeping
            # -m: Prevent disk from idle sleeping
            env = os.environ.copy()
            env['CAFFEINATED'] = '1'  # Mark that we're now caffeinated
            
            try:
                result = subprocess.run(
                    ['caffeinate', '-dims', sys.executable] + sys.argv,
                    env=env
                )
                sys.exit(result.returncode)
            except KeyboardInterrupt:
                print("\n\nTraining interrupted by user")
                sys.exit(130)
            except Exception as e:
                print(f"\n\nError starting caffeinate: {e}")
                print("Continuing without caffeinate...")


    main()
