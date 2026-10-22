"""One-time editor asset generation. Python is not used by the playable game."""
import unreal

ROOT = '/Game/TagArena'
MAP = ROOT + '/Maps/TagArena'
assets = unreal.EditorAssetLibrary
tools = unreal.AssetToolsHelpers.get_asset_tools()
actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
levels = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)

palette = {
    'Floor': (0.025, 0.041, 0.065),
    'Grid': (0.039, 0.061, 0.087),
    'Wall': (0.12, 0.18, 0.25),
    'Cover': (0.115, 0.13, 0.23),
    'CoverInset': (0.16, 0.18, 0.30),
    'Lane': (0.045, 0.105, 0.13),
    'LaneEdge': (0.13, 0.32, 0.37),
    'Runner': (0.03, 0.75, 0.95),
    'Tagger': (1.0, 0.16, 0.23),
    'Goal': (0.03, 0.30, 0.22),
    'GoalEdge': (0.12, 0.9, 0.55),
    'Commit': (0.95, 0.65, 0.16),
}
materials = {}
for name, rgb in palette.items():
    path = ROOT + '/Materials/M_' + name
    material = assets.load_asset(path) if assets.does_asset_exist(path) else None
    if not material:
        material = tools.create_asset('M_' + name, ROOT + '/Materials', unreal.Material, unreal.MaterialFactoryNew())
        material.set_editor_property('shading_model', unreal.MaterialShadingModel.MSM_UNLIT)
        color = unreal.MaterialEditingLibrary.create_material_expression(material, unreal.MaterialExpressionConstant3Vector)
        color.set_editor_property('constant', unreal.LinearColor(*rgb, 1.0))
        unreal.MaterialEditingLibrary.connect_material_property(color, '', unreal.MaterialProperty.MP_EMISSIVE_COLOR)
        unreal.MaterialEditingLibrary.recompile_material(material)
        assets.save_loaded_asset(material)
    materials[name] = material

if assets.does_asset_exist(MAP):
    unreal.log('TagArena map already exists; preserving the existing map.')
else:
    if not levels.new_level(MAP):
        raise RuntimeError('Could not create TagArena map')
    cube = assets.load_asset('/Engine/BasicShapes/Cube')
    cylinder = assets.load_asset('/Engine/BasicShapes/Cylinder')

    def box(label, location, size, material, collide=False, mesh=cube):
        actor = actors.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector(*location))
        actor.set_actor_label(label)
        component = actor.static_mesh_component
        component.set_static_mesh(mesh)
        component.set_material(0, materials[material])
        component.set_collision_enabled(unreal.CollisionEnabled.QUERY_AND_PHYSICS if collide else unreal.CollisionEnabled.NO_COLLISION)
        actor.set_actor_scale3d(unreal.Vector(*(v / 100.0 for v in size)))
        return actor

    box('Arena | Floor', (0, 0, -15), (1340, 1280, 30), 'Floor', True)
    for x in range(-600, 601, 100):
        box('Grid X ' + str(x), (x, 0, 0.5), (1.5, 1220, 1), 'Grid')
    for y in range(-600, 601, 100):
        box('Grid Y ' + str(y), (0, y, 0.5), (1300, 1.5, 1), 'Grid')
    for side in [-1, 1]:
        name = 'LEFT' if side < 0 else 'RIGHT'
        box(name + ' | Lane', (0, side * 500, 1.5), (670, 156, 1), 'Lane')
        for y in [side * 414, side * 586]:
            box(name + ' | Edge', (0, y, 2), (670, 3, 1), 'LaneEdge')
        for x in range(-280, 301, 65):
            box(name + ' | Guide', (x, side * 500, 2.5), (22, 3, 1), 'LaneEdge')
        box(name + ' | Commitment line', (-250, side * 500, 3), (7, 155, 1), 'Commit')
    box('Cover | Collision', (0, 0, 60), (440, 800, 120), 'Cover', True)
    box('Cover | Inset', (0, 0, 121), (390, 750, 2), 'CoverInset')
    box('Boundary | North', (660, 0, 40), (24, 1260, 80), 'Wall', True)
    box('Boundary | South', (-660, 0, 40), (24, 1260, 80), 'Wall', True)
    box('Boundary | Left', (0, -612, 40), (1300, 24, 80), 'Wall', True)
    box('Boundary | Right', (0, 612, 40), (1300, 24, 80), 'Wall', True)
    box('Escape zone', (570, 0, 2), (105, 1210, 2), 'Goal')
    box('Escape line', (515, 0, 4), (6, 1210, 2), 'GoalEdge')
    box('Runner start ring', (-500, 0, 3), (160, 160, 2), 'LaneEdge', mesh=cylinder)
    box('Runner start inset', (-500, 0, 5), (148, 148, 2), 'Floor', mesh=cylinder)

    camera = actors.spawn_actor_from_class(unreal.CameraActor, unreal.Vector(0, 0, 2400), unreal.Rotator(pitch=-90, yaw=0, roll=0))
    camera.set_actor_label('Arena | Top-down camera')
    camera.tags = ['TagArenaCamera']
    camera.camera_component.set_editor_property('projection_mode', unreal.CameraProjectionMode.ORTHOGRAPHIC)
    camera.camera_component.set_editor_property('ortho_width', 3200.0)
    camera.camera_component.set_editor_property('constrain_aspect_ratio', True)
    camera.camera_component.set_editor_property('aspect_ratio', 1.6)

    volume = actors.spawn_actor_from_class(unreal.PostProcessVolume, unreal.Vector(0, 0, 0))
    volume.set_actor_label('Arena | Stable exposure')
    volume.set_editor_property('unbound', True)
    settings = volume.get_editor_property('settings')
    settings.set_editor_property('override_auto_exposure_method', True)
    settings.set_editor_property('auto_exposure_method', unreal.AutoExposureMethod.AEM_MANUAL)
    settings.set_editor_property('override_auto_exposure_bias', True)
    settings.set_editor_property('auto_exposure_bias', 0.0)
    settings.set_editor_property('override_auto_exposure_apply_physical_camera_exposure', True)
    settings.set_editor_property('auto_exposure_apply_physical_camera_exposure', False)
    settings.set_editor_property('override_bloom_intensity', True)
    settings.set_editor_property('bloom_intensity', 0.0)
    settings.set_editor_property('override_vignette_intensity', True)
    settings.set_editor_property('vignette_intensity', 0.0)
    volume.set_editor_property('settings', settings)
    if not levels.save_current_level():
        raise RuntimeError('Could not save TagArena map')
    unreal.log('Created ' + MAP)
