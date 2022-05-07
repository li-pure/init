from __future__ import division

import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Video


class ObliqueSource():
    def __init__(self):
        self.resolution = 50 # pixels/μm
        self.cell_size = mp.Vector3(14, 14)
        self.pml_layers = [mp.PML(thickness=2)]

    def single_point(self):

        # rotation angle (in degrees) of waveguide, counter clockwise (CCW) around z-axis
        rot_angle = np.radians(20)

        geometry = [mp.Block(center=mp.Vector3(),
                             size=mp.Vector3(mp.inf, 1, mp.inf),
                             e1 = mp.Vector3(1).rotate(mp.Vector3(z=1), rot_angle),
                             e2 = mp.Vector3(y=1).rotate(mp.Vector3(z=1), rot_angle),
                             material=mp.Medium(epsilon=12))]

        fsrc = 0.15 # frequency of eigenmode or constant-amplitude source

        sources = [mp.Source(src=mp.GaussianSource(fsrc, fwidth=0.2*fsrc),
                                 center=mp.Vector3(),
                                 size=mp.Vector3(y=2),
                                 component=mp.Ez)]

        sim = mp.Simulation(cell_size=self.cell_size,
                            resolution=self.resolution,
                            boundary_layers=self.pml_layers,
                            sources=sources,
                            geometry=geometry)

        f = plt.figure(dpi=100)
        sim.plot2D(ax=f.gca())   # gca: get current Axes object 'ax'
        plt.show()

        # f = plt.figure(dpi=100)
        # animate = mp.Animate2D(sim, mp.Ez, f=f, normalize=True)
        # sim.run(mp.at_every(1, animate), until_after_sources=50)
        # plt.close()
        #
        # filename = 'media/oblique-source-normal.mp4'
        # animate.to_mp4(10, filename)
        # Video(filename)

    def multi_point(self):
        for rot_angle in np.radians([20]):

            geometry = [mp.Block(center=mp.Vector3(),
                                 size=mp.Vector3(mp.inf, 1, mp.inf),
                                 e1=mp.Vector3(1).rotate(mp.Vector3(z=1), rot_angle),
                                 e2=mp.Vector3(y=1).rotate(mp.Vector3(z=1), rot_angle),
                                 material=mp.Medium(epsilon=12))]

            fsrc = 0.15  # frequency of eigenmode or constant-amplitude source
            bnum = 1  # band number of eigenmode
            kpoint = mp.Vector3(x=1).rotate(mp.Vector3(z=1), rot_angle)

            sources = [mp.EigenModeSource(src=mp.GaussianSource(fsrc, fwidth=0.2 * fsrc),
                                          center=mp.Vector3(),
                                          size=mp.Vector3(y=14),
                                          direction=mp.NO_DIRECTION,
                                          eig_kpoint=kpoint,
                                          eig_band=bnum,
                                          eig_parity=mp.ODD_Z,
                                          eig_match_freq=True)]

            sim = mp.Simulation(cell_size=self.cell_size,
                                resolution=self.resolution,
                                boundary_layers=self.pml_layers,
                                sources=sources,
                                geometry=geometry)

            tran = sim.add_flux(fsrc, 0, 1, mp.FluxRegion(center=mp.Vector3(x=5), size=mp.Vector3(y=14)))
            sim.run(until_after_sources=50)
            res = sim.get_eigenmode_coefficients(tran,
                                                 [1],
                                                 eig_parity=mp.EVEN_Y + mp.ODD_Z if rot_angle == 0 else mp.ODD_Z,
                                                 direction=mp.NO_DIRECTION,
                                                 kpoint_func=lambda f, n: kpoint)
            print("flux:, {:.6f}, {:.6f}".format(mp.get_fluxes(tran)[0], abs(res.alpha[0, 0, 0]) ** 2))

            sim.plot2D(fields=mp.Ez)
            plt.show()

class wvg_src():
    def process(self):
        # Example file illustrating an eigenmode source, generating a waveguide mode
        # (requires recent MPB version to be installed before Meep is compiled)

        cell = mp.Vector3(16, 8)

        # an asymmetrical dielectric waveguide:
        geometry = [
            mp.Block(center=mp.Vector3(), size=mp.Vector3(mp.inf, 1, mp.inf),
                     material=mp.Medium(epsilon=12)),
            # hollow
            mp.Block(center=mp.Vector3(y=0.3), size=mp.Vector3(mp.inf, 0.1, mp.inf),
                     material=mp.Medium())
        ]

        # create a transparent source that excites a right-going waveguide mode
        sources = [
            mp.EigenModeSource(src=mp.ContinuousSource(0.15), size=mp.Vector3(y=6),
                               center=mp.Vector3(x=-5), component=mp.Dielectric,
                               eig_parity=mp.ODD_Z)
        ]

        pml_layers = [mp.PML(1.0)]

        force_complex_fields = True  # so we can get time-average flux

        resolution = 10

        sim = mp.Simulation(
            cell_size=cell,
            geometry=geometry,
            sources=sources,
            boundary_layers=pml_layers,
            force_complex_fields=force_complex_fields,
            resolution=resolution
        )

        f = plt.figure(dpi=100)
        sim.plot2D(ax=f.gca())  # gca: get current Axes object 'ax'
        plt.show()

        sim.run(
            mp.at_beginning(mp.output_epsilon),
            mp.at_end(mp.output_png(mp.Ez, "-a yarg -A $EPS -S3 -Zc dkbluered", rm_h5=False)),
            until=200
        )

        flux1 = sim.flux_in_box(mp.X, mp.Volume(center=mp.Vector3(-6.0), size=mp.Vector3(1.8, 6)))
        flux2 = sim.flux_in_box(mp.X, mp.Volume(center=mp.Vector3(6.0), size=mp.Vector3(1.8, 6)))

        # averaged over y region of width 1.8
        print("left-going flux = {}".format(flux1 / -1.8))

class StraightWvg():
    def __init__(self, eig_src, compute_flux):
        """
        eig_src = True  # eigenmode (True) or constant-amplitude (False) source
        compute_flux = True  # compute flux (True) or plot the field profile (False)
        :param eig_src:
        :param compute_flux:
        """
        self.eig_src = eig_src
        self.compute_flux = compute_flux

    def process(self):
        resolution = 50  # pixels/μm
        cell_size = mp.Vector3(17, 8)
        pml_layers = [mp.PML(thickness=1)]

        geometry = [mp.Block(center=mp.Vector3(),
                             size=mp.Vector3(mp.inf, 1, mp.inf),
                             material=mp.Medium(epsilon=12))]

        fsrc = 0.15  # frequency of eigenmode or constant-amplitude source
        bnum = 1  # band number of eigenmode
        kpoint = mp.Vector3(x=1)

        if self.eig_src:
            sources = [mp.EigenModeSource(
                src=mp.GaussianSource(fsrc, fwidth=0.2 * fsrc) if self.compute_flux else mp.ContinuousSource(fsrc),
                center=mp.Vector3(x=-7),
                size=mp.Vector3(y=3),
                direction=mp.NO_DIRECTION,
                eig_kpoint=kpoint,
                eig_band=bnum,
                eig_parity=mp.EVEN_Y + mp.ODD_Z,
                eig_match_freq=True)]
        else:
            sources = [mp.Source(src=mp.GaussianSource(fsrc, fwidth=0.2 * fsrc) if self.compute_flux else mp.ContinuousSource(fsrc),
                                 center=mp.Vector3(),
                                 size=mp.Vector3(y=3),
                                 component=mp.Ez)]

        # Simulation
        sim = mp.Simulation(cell_size=cell_size,
                            resolution=resolution,
                            boundary_layers=pml_layers,
                            sources=sources,
                            geometry=geometry,
                            symmetries=[mp.Mirror(mp.Y)])

        if self.compute_flux:
            tran = sim.add_flux(fsrc, 0, 1,
                                mp.FluxRegion(center=mp.Vector3(x=5), size=mp.Vector3(y=14)))  # Type: DftFlux
            sim.run(until_after_sources=50)
            res = sim.get_eigenmode_coefficients(tran,
                                                 [1],
                                                 eig_parity=mp.EVEN_Y + mp.ODD_Z,
                                                 # If removed: RuntimeError: meep: flux regions for eigenmode projection
                                                 # with symmetry should be created by add_mode_monitor()
                                                 direction=mp.NO_DIRECTION,
                                                 kpoint_func=lambda f, n: kpoint)
            print("flux:, {:.6f}, {:.6f}".format(mp.get_fluxes(tran)[0], abs(res.alpha[0, 0, 0]) ** 2))
            sim.plot2D(fields=mp.Ez)
            plt.show()

        else:
            sim.run(until=100)
            # sim.plot2D(output_plane=mp.Volume(center=mp.Vector3(), size=mp.Vector3(10, 10)),
            #            fields=mp.Ez,
            #            field_parameters={'alpha': 0.9})
            sim.plot2D(fields=mp.Ez)   # value is very small
            # plt.colorbar()
            plt.show()


class BendWvg():
    def __init__(self, eig_src, compute_flux):
        """
        eig_src = True  # eigenmode (True) or constant-amplitude (False) source
        compute_flux = True  # compute flux (True) or plot the field profile (False)
        :param eig_src:
        :param compute_flux:
        """
        self.eig_src = eig_src
        self.compute_flux = compute_flux

    def process(self):
        resolution = 20                        # pixels/μm
        cell_size = mp.Vector3(18, 18)
        pml_layers = [mp.PML(thickness=1)]

        geometry = [mp.Block(center=mp.Vector3(-2.75, -3.5, 0),
                             size=mp.Vector3(11.5, 1, mp.inf),
                             material=mp.Medium(epsilon=12)),
                    mp.Block(center=mp.Vector3(3.5, 2, 0),
                             size=mp.Vector3(1, 12, mp.inf),
                             material=mp.Medium(epsilon=12))]

        fsrc = 0.15  # frequency of eigenmode or constant-amplitude source
        bnum = 1  # band number of eigenmode
        kpoint = mp.Vector3(x=1)

        if self.eig_src:
            sources = [mp.EigenModeSource(
                src=mp.GaussianSource(fsrc, fwidth=0.2 * fsrc) if self.compute_flux else mp.ContinuousSource(fsrc),
                center=mp.Vector3(x=-7, y=-3.5),
                size=mp.Vector3(y=3),
                direction=mp.NO_DIRECTION,
                eig_kpoint=kpoint,
                eig_band=bnum,
                eig_parity=mp.ODD_Z,
                eig_match_freq=True)]
        else:
            sources = [mp.Source(src=mp.GaussianSource(fsrc, fwidth=0.2 * fsrc) if self.compute_flux else mp.ContinuousSource(fsrc),
                                 center=mp.Vector3(),
                                 size=mp.Vector3(y=3),
                                 component=mp.Ez)]

        # Simulation
        sim = mp.Simulation(cell_size=cell_size,
                            resolution=resolution,
                            boundary_layers=pml_layers,
                            sources=sources,
                            geometry=geometry)

        f = plt.figure(dpi=100)
        sim.plot2D(ax=f.gca())  # gca: get current Axes object 'ax'
        plt.show()

        if self.compute_flux:
            tran = sim.add_flux(fsrc, 0, 1,
                                mp.FluxRegion(center=mp.Vector3(x=5), size=mp.Vector3(y=14)))  # Type: DftFlux
            sim.run(until_after_sources=50)
            res = sim.get_eigenmode_coefficients(tran,
                                                 [1],
                                                 eig_parity=mp.ODD_Z,
                                                 # If removed: RuntimeError: meep: flux regions for eigenmode projection
                                                 # with symmetry should be created by add_mode_monitor()
                                                 direction=mp.NO_DIRECTION,
                                                 kpoint_func=lambda f, n: kpoint)
            print("flux:, {:.6f}, {:.6f}".format(mp.get_fluxes(tran)[0], abs(res.alpha[0, 0, 0]) ** 2))
            sim.plot2D(fields=mp.Ez)
            plt.show()

        else:
            sim.run(until=100)
            sim.plot2D(fields=mp.Ez)   # value is very small
            # plt.colorbar()
            plt.show()

if __name__ == '__main__':
    print('start')

    # objects = ObliqueSource()
    # objects.single_point()

    # objects = wvg_src()
    # objects.process()

    # # [1] my simulation : straight waveguide   --corresponding to lumerical
    # objects = StraightWvg(eig_src=True, compute_flux=False)
    # objects.process()

    # [2] my simulation : bend waveguide   --corresponding to lumerical
    objects = BendWvg(eig_src=True, compute_flux=False)
    objects.process()