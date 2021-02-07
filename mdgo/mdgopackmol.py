import subprocess


def run_packmol(packmol_input):
    """Run and check that Packmol worked correctly"""
    try:
        p = subprocess.run('packmol < {}'.format(packmol_input),
                           check=True,
                           shell=True,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise ValueError("Packmol failed with errorcode {}"
                         " and stderr: {}".format(e.returncode, e.stderr))
    else:
        with open('packmol.stdout', 'w') as out:
            out.write(p.stdout.decode())


def make_packmol_input(structures, numbers, box, input='packmol.inp',
                       output='output.xyz', tolerance=None, seed=None):
    """Convert the call into a Packmol usable input file
    Parameters
    ----------
    structures : list
      list of PackmolStructure objects
    tolerance : float, optional
      minimum distance between molecules, defaults to 2.0
    """

    if tolerance is None:
        tolerance = 2.0

    if seed is None:
        seed = 123

    with open(input, 'w') as out:
        out.write("# " + ' + '.join(numbers[structure["name"]] + " " + structure["name"]
                 for structure in structures) + "\n\n")

        out.write('seed {}\n'.format(seed))
        out.write('tolerance {}\n\n'.format(tolerance))
        out.write('filetype xyz\n\n')

        for structure in structures:
            out.write("structure {}\n".format(structure["file"]))
            out.write("  number {}\n".format(numbers[structure["name"]]))
            out.write("  inside box {}\n".format(" ".join(str(i) for i in box)))
            out.write("end structure\n\n")
        out.write('output {}\n\n'.format(output))


def main():
    structures = [{"name": "EMC",
                   "file": "/Users/th/Downloads/test_selenium/EMC.lmp.xyz"}]
    make_packmol_input(structures, {"EMC": '2'}, [0., 0., 0., 10., 10., 10.])
    run_packmol('packmol.inp')


if __name__ == "__main__":
    main()


