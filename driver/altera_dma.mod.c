#include <linux/module.h>
#include <linux/vermagic.h>
#include <linux/compiler.h>

MODULE_INFO(vermagic, VERMAGIC_STRING);

__visible struct module __this_module
__attribute__((section(".gnu.linkonce.this_module"))) = {
	.name = KBUILD_MODNAME,
	.init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
	.exit = cleanup_module,
#endif
	.arch = MODULE_ARCH_INIT,
};

#ifdef RETPOLINE
MODULE_INFO(retpoline, "Y");
#endif

static const struct modversion_info ____versions[]
__used
__attribute__((section("__versions"))) = {
	{ 0x59253af, __VMLINUX_SYMBOL_STR(module_layout) },
	{ 0x35b3df96, __VMLINUX_SYMBOL_STR(pci_unregister_driver) },
	{ 0x2e3cb253, __VMLINUX_SYMBOL_STR(__pci_register_driver) },
	{ 0x9c5bc552, __VMLINUX_SYMBOL_STR(finish_wait) },
	{ 0xcb128141, __VMLINUX_SYMBOL_STR(prepare_to_wait_event) },
	{ 0x1000e51, __VMLINUX_SYMBOL_STR(schedule) },
	{ 0x622598b1, __VMLINUX_SYMBOL_STR(init_wait_entry) },
	{ 0xb35dea8f, __VMLINUX_SYMBOL_STR(__arch_copy_to_user) },
	{ 0x79aa04a2, __VMLINUX_SYMBOL_STR(get_random_bytes) },
	{ 0x65345022, __VMLINUX_SYMBOL_STR(__wake_up) },
	{ 0x4f68e5c9, __VMLINUX_SYMBOL_STR(do_gettimeofday) },
	{ 0xdcb764ad, __VMLINUX_SYMBOL_STR(memset) },
	{ 0x84bc974b, __VMLINUX_SYMBOL_STR(__arch_copy_from_user) },
	{ 0x88db9f48, __VMLINUX_SYMBOL_STR(__check_object_size) },
	{ 0xd2b09ce5, __VMLINUX_SYMBOL_STR(__kmalloc) },
	{ 0xeae3dfd6, __VMLINUX_SYMBOL_STR(__const_udelay) },
	{ 0x2a098b39, __VMLINUX_SYMBOL_STR(kmalloc_caches) },
	{ 0x9b63897d, __VMLINUX_SYMBOL_STR(dma_alloc_from_coherent_attr) },
	{ 0xab40cca9, __VMLINUX_SYMBOL_STR(__init_waitqueue_head) },
	{ 0xf24b3dfe, __VMLINUX_SYMBOL_STR(__ioremap) },
	{ 0x4fcf516a, __VMLINUX_SYMBOL_STR(pci_bus_read_config_byte) },
	{ 0xa9be000e, __VMLINUX_SYMBOL_STR(pci_enable_msi_range) },
	{ 0x8510ce42, __VMLINUX_SYMBOL_STR(pci_set_master) },
	{ 0x31d1f902, __VMLINUX_SYMBOL_STR(pci_request_regions) },
	{ 0x3ecd7d27, __VMLINUX_SYMBOL_STR(_dev_info) },
	{ 0xc920433a, __VMLINUX_SYMBOL_STR(pci_enable_device) },
	{ 0xea421b3b, __VMLINUX_SYMBOL_STR(dev_err) },
	{ 0x7856fdb0, __VMLINUX_SYMBOL_STR(cdev_add) },
	{ 0xd49c3a37, __VMLINUX_SYMBOL_STR(cdev_init) },
	{ 0x29537c9e, __VMLINUX_SYMBOL_STR(alloc_chrdev_region) },
	{ 0x81f3f299, __VMLINUX_SYMBOL_STR(kmem_cache_alloc_trace) },
	{ 0x328689d4, __VMLINUX_SYMBOL_STR(dma_ops) },
	{ 0x37a0cba, __VMLINUX_SYMBOL_STR(kfree) },
	{ 0x28131c43, __VMLINUX_SYMBOL_STR(dma_release_from_coherent_attr) },
	{ 0xf20dabd8, __VMLINUX_SYMBOL_STR(free_irq) },
	{ 0x27e1a049, __VMLINUX_SYMBOL_STR(printk) },
	{ 0xd8688729, __VMLINUX_SYMBOL_STR(pci_release_regions) },
	{ 0x181f9bf2, __VMLINUX_SYMBOL_STR(pci_disable_msi) },
	{ 0xc3782eb8, __VMLINUX_SYMBOL_STR(pci_disable_device) },
	{ 0x7485e15e, __VMLINUX_SYMBOL_STR(unregister_chrdev_region) },
	{ 0xb6fcec34, __VMLINUX_SYMBOL_STR(cdev_del) },
	{ 0x1fdc7df2, __VMLINUX_SYMBOL_STR(_mcount) },
};

static const char __module_depends[]
__used
__attribute__((section(".modinfo"))) =
"depends=";

MODULE_ALIAS("pci:v00001172d0000E003sv*sd*bc*sc*i*");

MODULE_INFO(srcversion, "2F0695E7A37CA1670A8A8D1");
